/**
 * The contour inventory: what was extracted, what is going into the model, and a way to
 * change your mind about any of it.
 *
 * The screen this replaces said `166 of 166 contours · 7365 of 7365 points` and nothing
 * else — a count with no way to look at what it counted. `teach_preview` has always
 * returned the two facts that separate the part from the bench it is sitting on, arc
 * `length` and `mean_strength`, and neither reached the screen; there was no order, no
 * filter, no way to select several, and no way to step through them one at a time.
 *
 * Hovering a row lights the contour on the canvas and vice versa, because a list of a
 * hundred and sixty-six numbered things is only useful if you can see which one is which.
 */

import {
  Badge,
  Button,
  Panel,
  SegmentedControl,
  Select,
  Table,
  type Column,
} from "@vitavision/lab-ui";
import { Crosshair } from "lucide-react";

import type { ContourStat, KeepFilter, SelectMode, SortKey } from "../../canvas/contourSelection";

/** Short enough not to wrap in a narrow inspector; the column headers say what they mean. */
const SORTS: { value: SortKey; label: string }[] = [
  { value: "length", label: "Longest" },
  { value: "strength", label: "Strongest" },
  { value: "points", label: "Most points" },
  { value: "id", label: "By index" },
];

const FILTERS: { value: KeepFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "kept", label: "Kept" },
  { value: "dropped", label: "Dropped" },
];

export function ContourSection({
  stats,
  visible,
  kept,
  selected,
  hovered,
  sort,
  onSort,
  filter,
  onFilter,
  keptPoints,
  totalPoints,
  selectedPoints,
  onSelect,
  onKeep,
  onHover,
  onKeepAll,
  onDropAll,
  onInvert,
  onFrame,
  follow,
  onFollow,
}: {
  stats: ContourStat[];
  visible: ContourStat[];
  kept: ReadonlySet<number>;
  selected: ReadonlySet<number>;
  hovered: number | null;
  sort: SortKey;
  onSort: (sort: SortKey) => void;
  filter: KeepFilter;
  onFilter: (filter: KeepFilter) => void;
  keptPoints: number;
  totalPoints: number;
  selectedPoints: number;
  onSelect: (ids: number[], mode: SelectMode) => void;
  onKeep: (ids: number[], keep: boolean) => void;
  onHover: (id: number | null) => void;
  onKeepAll: () => void;
  onDropAll: () => void;
  onInvert: () => void;
  onFrame: () => void;
  follow: boolean;
  onFollow: (on: boolean) => void;
}) {
  const maxStrength = Math.max(...stats.map((stat) => stat.strength), 1e-6);

  const columns: Column<ContourStat>[] = [
    {
      key: "id",
      header: "#",
      width: "2.6rem",
      cell: (stat) => (
        <span className="flex items-center gap-1 font-mono tabular-nums">
          {stat.id}
          {stat.closed && (
            <span
              aria-label="closed"
              title="Closed contour"
              className="size-1.5 rounded-full bg-fg-subtle"
            />
          )}
        </span>
      ),
    },
    {
      key: "strength",
      header: "strength",
      width: "5.5rem",
      cell: (stat) => (
        <span className="flex items-center gap-1.5">
          <span className="h-1 flex-1 rounded-full bg-line">
            <span
              className="block h-full rounded-full bg-signal"
              style={{ width: `${Math.max(2, (100 * stat.strength) / maxStrength)}%` }}
            />
          </span>
          <span className="font-mono tabular-nums text-fg-muted">{stat.strength.toFixed(2)}</span>
        </span>
      ),
    },
    { key: "length", header: "len", numeric: true, cell: (stat) => stat.length.toFixed(0) },
    { key: "points", header: "pts", numeric: true, cell: (stat) => stat.points },
    {
      key: "keep",
      header: "keep",
      width: "2.2rem",
      cell: (stat) => (
        <input
          type="checkbox"
          checked={kept.has(stat.id)}
          aria-label={`Keep contour ${stat.id}`}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => onKeep([stat.id], event.target.checked)}
          className="size-3.5 accent-[var(--signal)]"
        />
      ),
    },
  ];

  return (
    <Panel
      title="Contours"
      actions={
        <span className="font-mono text-[10px] text-fg-subtle tabular-nums">
          {visible.length === stats.length ? stats.length : `${visible.length}/${stats.length}`}
        </span>
      }
      bodyClassName="p-0"
    >
      <div className="flex flex-col gap-2 p-2.5 pb-2">
        <div>
          <div className="flex items-baseline justify-between gap-2 text-[11px]">
            <span className="text-fg">
              <Badge tone="info">{kept.size}</Badge> kept of {stats.length}
            </span>
            <span className="font-mono text-fg-muted tabular-nums">
              {keptPoints.toLocaleString()} / {totalPoints.toLocaleString()} pts
            </span>
          </div>
          <span className="mt-1 block h-1 rounded-full bg-line">
            <span
              className="block h-full rounded-full bg-signal transition-[width]"
              style={{ width: `${totalPoints > 0 ? (100 * keptPoints) / totalPoints : 0}%` }}
            />
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <SegmentedControl
            value={filter}
            options={FILTERS}
            onValueChange={(value) => onFilter(value as KeepFilter)}
            className="flex-1"
          />
          <Select
            value={sort}
            options={SORTS}
            onValueChange={(value) => onSort(value as SortKey)}
            aria-label="Sort contours"
            className="w-32"
          />
        </div>

        <div className="flex items-center gap-1">
          <Button variant="ghost" onClick={onKeepAll}>
            Keep all
          </Button>
          <Button variant="ghost" onClick={onDropAll}>
            Drop all
          </Button>
          <Button variant="ghost" onClick={onInvert}>
            Invert
          </Button>
        </div>
      </div>

      {/* Capped so the inventory scrolls inside the column instead of pushing the datum and
          the build button off the bottom of it. */}
      <div className="max-h-[42vh] overflow-y-auto border-y border-line px-2.5">
        <Table
          columns={columns}
          rows={visible}
          rowKey={(stat) => stat.id}
          caption="Candidate contours"
          empty="No contours match this filter."
          isRowActive={(stat) => selected.has(stat.id) || hovered === stat.id}
          onRowHover={(stat) => onHover(stat?.id ?? null)}
          onRowClick={(stat, _index, event) =>
            onSelect([stat.id], modeFor(event as { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean }))
          }
        />
      </div>

      <div className="flex flex-wrap items-center gap-1 p-2.5 pt-2">
        {selected.size === 0 ? (
          <p className="text-[11px] text-fg-subtle">
            Click a contour — on the image or in the list — to work with it. Shift-drag on the
            image sweeps several; ↑/↓ steps through them.
          </p>
        ) : (
          <>
            <span className="mr-auto font-mono text-[11px] text-fg tabular-nums">
              {selected.size} selected · {selectedPoints.toLocaleString()} pts
            </span>
            <Button variant="ghost" icon={<Crosshair />} onClick={onFrame} title="Zoom to the selection (F)">
              Frame
            </Button>
            <Button variant="ghost" onClick={() => onKeep([...selected], true)}>
              Keep
            </Button>
            <Button variant="danger" onClick={() => onKeep([...selected], false)}>
              Drop
            </Button>
          </>
        )}
        <label className="flex w-full items-center gap-1.5 text-[10px] text-fg-subtle">
          <input
            type="checkbox"
            checked={follow}
            onChange={(event) => onFollow(event.target.checked)}
            className="size-3 accent-[var(--signal)]"
          />
          Follow the selection with the canvas
        </label>
      </div>
    </Panel>
  );
}

function modeFor(event: { metaKey: boolean; ctrlKey: boolean; shiftKey: boolean }): SelectMode {
  if (event.metaKey || event.ctrlKey) return "toggle";
  if (event.shiftKey) return "range";
  return "replace";
}
