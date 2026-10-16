"""Cross-section annotation on a local page: two boundaries per station, twice, blind.

It proposes stations on a curve per frame (the manifest's seed polyline, or `--curves`, a
tracked curve per frame) and serves one task at a time on loopback: the frame crop around
the station and the magnified profile along the normal, sampled every 0.05 px. Click the
profile to place the active boundary; the keys nudge it. Each section comes back once in
pass 2, in another shuffled order and with no earlier mark shown. Every answer is written
to the annotations file at once, so a session resumes where it stopped.

    python tools/registration_eval/annotate.py --manifest seq/sequence.json --annotator va \\
        [--curves curves.json] [--count 60] [--seed 0] [--span FRAME,S0,S1,COUNT]

The annotations file defaults to the one the manifest names, or `annotations.json` beside
the manifest, so that annotations of private data stay with the data. Logic lives in
`annotation.py`; this file is the page and its loopback server.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import annotation
from manifest import Sequence
from viewer import png

CROP_RADIUS = 24


class App:
    """The page's state and answers, with no HTTP: the tests call it directly."""

    def __init__(self, sequence, session, annotator, curves, half_length=annotation.HALF_LENGTH,
                 method='cubic', sigma=1., crop_radius=CROP_RADIUS):
        self.sequence, self.session, self.annotator, self.curves = sequence, session, annotator, curves
        self.half_length, self.method, self.sigma, self.crop_radius = half_length, method, sigma, crop_radius

    def task(self):
        """The current task: geometry, profile and crop, never an earlier mark."""
        done, total = self.session.progress(self.annotator)
        current = self.session.next(self.annotator)
        if current is None:
            return {'done': True, 'progress': [done, total]}
        index, pass_ = current
        entry = self.session.sections[index]
        image, visible = self.sequence.image(entry['frame']), self.sequence.visible(entry['frame'])
        p = annotation.profile(image, visible, entry['station'], entry['normal'], self.half_length,
                               annotation.STEP, self.method, self.sigma)
        pixels, origin = annotation.crop(image, entry['station'], self.crop_radius)

        def finite(values):
            return [None if not np.isfinite(v) else round(float(v), 4) for v in values]
        return {'done': False, 'task': f'{index}.{pass_}', 'progress': [done, total],
                'frame': entry['frame'], 'station': entry['station'], 'normal': entry['normal'],
                'offsets': [round(float(t), 4) for t in p['offsets']],
                'values': finite(p['values']), 'derivative': finite(p['derivative']),
                'crop': png(pixels), 'origin': list(origin), 'method': self.method}

    def submit(self, body):
        index, pass_ = (int(v) for v in str(body['task']).split('.'))
        self.session.annotate(self.annotator, index, pass_, body.get('left'), body.get('right'))
        return self.task()

    def overview(self, frame):
        """A frame with its curve and stations, for adding a station."""
        if frame not in self.curves:
            raise ValueError(f'frame {frame} has no curve')
        image = self.sequence.image(frame).astype(float)
        lo, hi = np.percentile(image, (1, 99))
        pixels = np.rint(np.clip((image - lo)/max(hi - lo, 1e-9), 0, 1)*255).astype(np.uint8)
        return {'frame': frame, 'image': png(pixels), 'curve': self.curves[frame],
                'stations': [s['station'] for s in self.session.sections if s['frame'] == frame],
                'frames': sorted(self.curves)}

    def add_station(self, body):
        """Add a station at the curve point nearest to the clicked point."""
        frame = int(body['frame'])
        if frame not in self.curves:
            raise ValueError(f'frame {frame} has no curve')
        s = annotation.project(self.curves[frame], body['point'])
        added = self.session.add(annotation.section(frame, self.curves[frame], s, 'user'))
        return {'added': added, 'progress': list(self.session.progress(self.annotator)),
                **self.overview(frame)}


class Handler(BaseHTTPRequestHandler):
    app = None

    def log_message(self, *args):
        pass

    def reply(self, status, body, kind='application/json'):
        data = body.encode() if isinstance(body, str) else json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def answer(self, call):
        try:
            self.reply(200, call())
        except (ValueError, KeyError, TypeError) as error:
            self.reply(400, {'error': str(error)})

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            self.reply(200, PAGE, 'text/html; charset=utf-8')
        elif url.path == '/api/task':
            self.answer(self.app.task)
        elif url.path == '/api/overview':
            frame = parse_qs(url.query).get('frame', [None])[0]
            self.answer(lambda: self.app.overview(int(frame) if frame is not None else min(self.app.curves)))
        else:
            self.reply(404, {'error': 'not found'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except json.JSONDecodeError:
            return self.reply(400, {'error': 'invalid JSON'})
        routes = {'/api/annotate': self.app.submit, '/api/station': self.app.add_station}
        if self.path not in routes:
            return self.reply(404, {'error': 'not found'})
        self.answer(lambda: routes[self.path](body))


def server(app, port=0):
    """A loopback server for `app`; port 0 picks a free one."""
    handler = type('BoundHandler', (Handler,), {'app': app})
    return ThreadingHTTPServer(('127.0.0.1', port), handler)


def default_path(sequence):
    return sequence.annotations_path or sequence.root / 'annotations.json'


def open_session(sequence, path, curves, count, seed, spans, protocol):
    """Resume the annotations file at `path`, or create it with fresh stations."""
    if Path(path).exists():
        session = annotation.Session.open(path, len(sequence))
        for frame, s0, s1, n in spans:
            for entry in annotation.span_sections(curves, frame, s0, s1, n, session.seed):
                session.add(entry)
        return session
    sections = annotation.uniform_sections(curves, count, seed)
    for frame, s0, s1, n in spans:
        sections += annotation.span_sections(curves, frame, s0, s1, n, seed)
    return annotation.Session.create(path, sections, {'seed': seed, 'count': count, **protocol})


def parse_span(text):
    frame, s0, s1, count = text.split(',')
    return int(frame), float(s0), float(s1), int(count)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--annotator', required=True)
    parser.add_argument('--curves', type=Path, help='a curve per frame; the seed polyline by default')
    parser.add_argument('--annotations', type=Path, help='the annotations file to write or resume')
    parser.add_argument('--count', type=int, default=60, help='uniform stations over all curves')
    parser.add_argument('--seed', type=int, default=0, help='station and order seed')
    parser.add_argument('--span', type=parse_span, action='append', default=[],
                        help='FRAME,S0,S1,COUNT: extra stations on a hard span (arc length, px)')
    parser.add_argument('--half-length', type=float, default=annotation.HALF_LENGTH,
                        help='profile reach either side of the station, px')
    parser.add_argument('--profile', choices=('cubic', 'sampler'), default='cubic')
    parser.add_argument('--sigma', type=float, default=1., help='ProfileSampler σ, px')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if args.count < 0 or args.half_length <= 0 or args.sigma <= 0:
        parser.error('count >= 0, half-length > 0 and sigma > 0')
    sequence = Sequence(args.manifest)
    curves = annotation.load_curves(sequence, args.curves)
    path = args.annotations or default_path(sequence)
    protocol = {'curves': args.curves.name if args.curves else 'seed', 'profile': args.profile,
                'step_px': annotation.STEP, 'half_length_px': args.half_length}
    session = open_session(sequence, path, curves, args.count, args.seed, args.span, protocol)
    app = App(sequence, session, args.annotator, curves, args.half_length, args.profile, args.sigma)
    httpd = server(app, args.port)
    print(f'{path}: {len(session.sections)} sections; serving http://127.0.0.1:{httpd.server_port}/')
    if sequence.annotations_path is None:
        print(f'reference it from the manifest: "annotations": "{Path(path).name}"')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = (Path(__file__).with_name('annotate.html')).read_text()

if __name__ == '__main__':
    main()
