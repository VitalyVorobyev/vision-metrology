/**
 * What the model actually came out as.
 *
 * The old success message was one sentence that scrolled away, and the contours vanished
 * with it — so the one question worth asking after a build, "is this what I picked?", had no
 * answer on screen. The contours stay drawn now, and the model's own points are a layer over
 * them; this panel is the numbers beside that comparison.
 */

import { Button, Panel } from "@vitavision/lab-ui";
import { ArrowRight } from "lucide-react";

import type { ModelOut } from "../../api/backend";

export function ModelSection({
  model,
  onFind,
  onRebuild,
  rebuilding,
}: {
  model: ModelOut;
  onFind: () => void;
  onRebuild: () => void;
  rebuilding: boolean;
}) {
  const total = model.point_counts.reduce((sum, count) => sum + count, 0);
  const max = Math.max(...model.point_counts, 1);

  return (
    <Panel
      title={model.id}
      actions={
        <span className="font-mono text-[10px] text-fg-subtle tabular-nums">
          {model.num_levels_built} levels · {total.toLocaleString()} pts
        </span>
      }
    >
      <div className="flex flex-col gap-2">
        {/* Per level, because the pyramid is where a model quietly fails: a top level with
            a handful of points is a model the matcher cannot find anything with. */}
        <ol className="flex flex-col gap-0.5">
          {model.point_counts.map((count, level) => (
            <li key={level} className="flex items-center gap-1.5 font-mono text-[10px] tabular-nums">
              <span className="w-8 text-fg-subtle">L{level}</span>
              <span className="h-1.5 flex-1 rounded-full bg-line">
                <span
                  className="block h-full rounded-full bg-signal"
                  style={{ width: `${Math.max(2, (100 * count) / max)}%` }}
                />
              </span>
              <span className="w-10 text-right text-fg-muted">{count}</span>
            </li>
          ))}
        </ol>

        <p className="font-mono text-[10px] text-fg-subtle tabular-nums">
          origin ({model.origin[0].toFixed(1)}, {model.origin[1].toFixed(1)})
        </p>

        <div className="flex items-center gap-1">
          <Button variant="ghost" loading={rebuilding} onClick={onRebuild}>
            Rebuild
          </Button>
          <Button variant="primary" className="ml-auto" icon={<ArrowRight />} onClick={onFind}>
            Find it elsewhere
          </Button>
        </div>
      </div>
    </Panel>
  );
}
