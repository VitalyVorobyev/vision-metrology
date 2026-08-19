// Mirrors lab/backend/src/vm_lab/schemas.py. Hand-kept in step with it — no codegen for
// an MVP this size; see lab/README.md for why.

export type Roi = [x: number, y: number, w: number, h: number];
export type AngleRange = [start: number, end: number];

export interface ImageOut {
  id: string;
  filename: string;
  width: number;
  height: number;
  sha256: string;
}

export interface ModelCreateRequest {
  image_id: string;
  roi: Roi;
  min_contrast: number;
  num_levels: number | null;
}

export interface ModelOut {
  id: string;
  image_id: string;
  roi: Roi;
  min_contrast: number;
  num_levels: number | null;
  origin: [number, number];
  num_levels_built: number;
  point_counts: number[];
}

export interface FindRequest {
  image_id: string;
  model_id: string;
  min_score: number;
  max_matches: number | null;
  roi: Roi | null;
  angle_range: AngleRange | null;
}

export interface MatchOut {
  x: number;
  y: number;
  angle: number;
  scale: number;
  score: number;
  support: number;
  level: number;
}

export interface FindResponse {
  matches: MatchOut[];
}

export type MeasureShapeKind = "circle" | "line";

export interface MeasureConfigIn {
  sigma?: number | null;
  threshold?: number | null;
  polarity?: "bright_to_dark" | "dark_to_bright" | "either" | null;
  max_obliquity_deg?: number | null;
}

export interface FitConfigIn {
  loss?: "l2" | "huber" | "tukey" | null;
  inlier_tol?: number | null;
}

export interface MeasureObjectIn {
  kind: MeasureShapeKind;
  label?: string | null;
  cx?: number | null;
  cy?: number | null;
  r?: number | null;
  arc?: AngleRange | null;
  ax?: number | null;
  ay?: number | null;
  bx?: number | null;
  by?: number | null;
  n_calipers: number;
  caliper_len: number;
  caliper_width: number;
  measure?: MeasureConfigIn;
  fit?: FitConfigIn;
}

export interface FixtureIn {
  x: number;
  y: number;
  angle: number;
  scale: number;
}

export interface MeasureRequest {
  image_id: string;
  model_id: string;
  fixture?: FixtureIn | null;
  min_score: number;
  objects: MeasureObjectIn[];
}

export interface EdgeMarkOut {
  pos_px: number;
  polarity: string;
}

export interface CaliperProfileOut {
  values: number[];
  step_px: number;
  edges: EdgeMarkOut[];
}

export interface CaliperResultOut {
  index: number;
  status: "hit" | "rejected";
  reason?: string | null;
  profile: CaliperProfileOut;
}

export interface OverlayPrimitiveOut {
  kind: "point" | "segment" | "circle" | "arc" | "caliper" | "dimension";
  tone?: "signal" | "normal" | "defect" | "warn" | "muted" | null;
  x?: number | null;
  y?: number | null;
  x1?: number | null;
  y1?: number | null;
  x2?: number | null;
  y2?: number | null;
  cx?: number | null;
  cy?: number | null;
  r?: number | null;
  startAngle?: number | null;
  endAngle?: number | null;
  width?: number | null;
  height?: number | null;
  angle?: number | null;
  label?: string | null;
  dashed?: boolean | null;
  cross?: boolean | null;
}

export interface MeasureObjectResultOut {
  kind: MeasureShapeKind | "error";
  label?: string | null;
  circle_cx?: number | null;
  circle_cy?: number | null;
  circle_r?: number | null;
  line_px?: number | null;
  line_py?: number | null;
  line_dx?: number | null;
  line_dy?: number | null;
  rms?: number | null;
  max_dev?: number | null;
  n_used?: number | null;
  message?: string | null;
  calipers: CaliperResultOut[];
  overlay: OverlayPrimitiveOut[];
}

export interface MeasureResponse {
  fixture: FixtureIn;
  fixture_source: "explicit" | "auto_find";
  objects: MeasureObjectResultOut[];
}
