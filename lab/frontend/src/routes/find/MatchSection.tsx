/**
 * The match inventory: what the search found on this frame, as a list you can work through.
 *
 * Each row is one match, named by its position in the search result. Hovering a row lights
 * the match on the canvas and hovering a match lights its row; a click selects it, and the
 * selection is the instance Verify compares. `↑`/`↓` step in the list's own order, `F`
 * frames the selection and `Esc` clears it (bound by the page, at the window).
 */

import { Badge, Button, Kbd, Panel, Table, cn, focusRing, type Column } from "@vitavision/ui";
import { ArrowDown, ArrowUp, Crosshair, ScanSearch } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ReactNode } from "react";

import type { MatchSort, MatchSortKey, MatchStat } from "../../state/matchInventory";

export function MatchSection({
  rows,
  total,
  sort,
  onSort,
  selected,
  hovered,
  onSelect,
  onHover,
  onFrame,
  onVerify,
  empty,
}: {
  /** The matches, in the list's order. */
  rows: MatchStat[];
  total: number;
  sort: MatchSort;
  onSort: (key: MatchSortKey) => void;
  selected: number | null;
  hovered: number | null;
  onSelect: (index: number) => void;
  onHover: (index: number | null) => void;
  onFrame: () => void;
  onVerify: () => void;
  /** What an empty list says: no search yet, nothing above the threshold, or a batch's miss. */
  empty: ReactNode;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const current = selected === null ? undefined : rows.find((row) => row.index === selected);

  // Keep the selection in view as the keys step through a list longer than its box.
  const position = rows.findIndex((row) => row.index === selected);
  useEffect(() => {
    if (position < 0) return;
    const row = scrollRef.current?.querySelectorAll("tbody tr")[position];
    row?.scrollIntoView?.({ block: "nearest" });
  }, [position]);

  const header = (key: MatchSortKey, label: string) => (
    <SortHeader label={label} active={sort.key === key} descending={sort.descending} onClick={() => onSort(key)} />
  );

  const columns: Column<MatchStat>[] = [
    {
      key: "index",
      header: header("index", "#"),
      width: "2.2rem",
      cell: (row) => <span className="font-mono tabular-nums">{row.index}</span>,
    },
    {
      key: "score",
      header: header("score", "score"),
      width: "5.6rem",
      cell: (row) => (
        <span className="flex items-center gap-1.5">
          <span className="h-1 flex-1 rounded-full bg-line">
            <span
              className="block h-full rounded-full bg-signal"
              style={{ width: `${Math.max(2, 100 * row.score)}%` }}
            />
          </span>
          <span className="font-mono tabular-nums text-fg">{row.score.toFixed(3)}</span>
        </span>
      ),
    },
    { key: "x", header: header("x", "x"), numeric: true, cell: (row) => row.x.toFixed(1) },
    { key: "y", header: header("y", "y"), numeric: true, cell: (row) => row.y.toFixed(1) },
    { key: "angle", header: header("angle", "angle°"), numeric: true, cell: (row) => row.angle.toFixed(1) },
    { key: "scale", header: header("scale", "scale"), numeric: true, cell: (row) => row.scale.toFixed(3) },
    { key: "support", header: header("support", "supp"), numeric: true, cell: (row) => row.support },
  ];

  return (
    <Panel
      title="Matches"
      actions={<span className="font-mono text-[10px] text-fg-subtle tabular-nums">{total}</span>}
      bodyClassName="p-0"
    >
      {/* Capped so a search with many instances scrolls inside the column. */}
      <div ref={scrollRef} className="max-h-[40vh] overflow-y-auto border-b border-line px-2.5">
        <Table
          columns={columns}
          rows={rows}
          rowKey={(row) => row.index}
          caption="Matches on this frame"
          empty={empty}
          isRowActive={(row) => row.index === selected || row.index === hovered}
          onRowHover={(row) => onHover(row?.index ?? null)}
          onRowClick={(row) => onSelect(row.index)}
        />
      </div>

      <div className="flex flex-wrap items-center gap-1 p-2.5 pt-2">
        {current === undefined ? (
          <p className="text-[11px] text-fg-subtle">
            Click a match, in the list or on the image. <Kbd>↑</Kbd> <Kbd>↓</Kbd> step,{" "}
            <Kbd>F</Kbd> frames, <Kbd>Esc</Kbd> clears.
          </p>
        ) : (
          <>
            <span className="mr-auto flex items-center gap-1.5 font-mono text-[11px] text-fg tabular-nums">
              <Badge tone="info">#{current.index}</Badge>
              score {current.score.toFixed(3)}
            </span>
            <Button variant="ghost" icon={<Crosshair />} onClick={onFrame} title="Zoom to the match (F)">
              Frame
            </Button>
            <Button variant="ghost" icon={<ScanSearch />} onClick={onVerify} title="Compare it with the model">
              Verify
            </Button>
          </>
        )}
      </div>
    </Panel>
  );
}

/**
 * A column header that sorts by its column. The ui `Table` has no sortable columns, so the
 * header is a button and says which way the column is sorted.
 */
function SortHeader({
  label,
  active,
  descending,
  onClick,
}: {
  label: string;
  active: boolean;
  descending: boolean;
  onClick: () => void;
}) {
  const Arrow = descending ? ArrowDown : ArrowUp;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Sort by ${label}${active ? (descending ? ", descending" : ", ascending") : ""}`}
      aria-pressed={active}
      className={cn(
        "inline-flex items-center gap-0.5 rounded-control hover:text-fg",
        active ? "text-fg" : "text-fg-muted",
        focusRing,
      )}
    >
      {label}
      {active && <Arrow className="size-2.5" aria-hidden />}
    </button>
  );
}
