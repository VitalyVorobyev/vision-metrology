/**
 * The caliper inventory: every caliper of every measured object, as a list you can work
 * through, and the selected caliper's profile.
 *
 * A fit's `rms` says how well the hits agree; it does not say which caliper missed, why, or
 * which hit pulled the fit. Each row here is one caliper: its verdict (a hit, or the reason
 * it was rejected), where along it the edge sat, that edge's residual against the object's
 * fit, and its amplitude. Hovering a row lights the caliper's box and edge mark on the
 * canvas and vice versa; the selected caliper's intensity profile is drawn below the list.
 */

import { LineProfile } from "@vitavision/charts";
import { Kbd, Panel, SegmentedControl, Table, cn, type Column } from "@vitavision/ui";
import { useEffect, useRef } from "react";

import {
  caliperProfile,
  reasonText,
  type CaliperFilter,
  type CaliperRow,
} from "../state/caliperInventory";

export function CaliperSection({
  rows,
  counts,
  filter,
  onFilter,
  selected,
  hovered,
  onSelect,
  onHover,
}: {
  /** The calipers the filter shows, in object then caliper order. */
  rows: CaliperRow[];
  counts: Record<CaliperFilter, number>;
  filter: CaliperFilter;
  onFilter: (filter: CaliperFilter) => void;
  selected: string | null;
  hovered: string | null;
  onSelect: (id: string) => void;
  onHover: (id: string | null) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the selection in view as the keys step through a list longer than its box.
  const position = rows.findIndex((row) => row.id === selected);
  useEffect(() => {
    if (position < 0) return;
    const row = scrollRef.current?.querySelectorAll("tbody tr")[position];
    row?.scrollIntoView?.({ block: "nearest" });
  }, [position]);

  const columns: Column<CaliperRow>[] = [
    {
      key: "id",
      header: "#",
      width: "2.6rem",
      cell: (row) => (
        <span className="font-mono tabular-nums">
          {row.object}.{row.index}
        </span>
      ),
    },
    {
      key: "result",
      header: "result",
      cell: (row) =>
        row.status === "hit" ? (
          <span className="text-signal">hit</span>
        ) : (
          <span className="text-defect">{reasonText(row.reason)}</span>
        ),
    },
    { key: "edge", header: "edge px", numeric: true, cell: (row) => fixed(row.edge, 2) },
    {
      key: "residual",
      header: "resid px",
      numeric: true,
      cell: (row) => fixed(row.residual, 3),
    },
    { key: "amplitude", header: "amp", numeric: true, cell: (row) => fixed(row.amplitude, 1) },
  ];

  return (
    <Panel
      title="Calipers"
      actions={<span className="font-mono text-[10px] text-fg-subtle tabular-nums">{counts.all}</span>}
      bodyClassName="p-0"
    >
      <div className="p-2.5 pb-2">
        <SegmentedControl
          value={filter}
          options={[
            { value: "all", label: `All ${counts.all}` },
            { value: "hits", label: `Hits ${counts.hits}` },
            { value: "rejected", label: `Rejected ${counts.rejected}` },
          ]}
          onValueChange={(value) => onFilter((value || "all") as CaliperFilter)}
          aria-label="Show calipers"
        />
      </div>

      {/* Capped so a model with many calipers scrolls inside the column. */}
      <div ref={scrollRef} className="max-h-[36vh] overflow-y-auto border-y border-line px-2.5">
        <Table
          columns={columns}
          rows={rows}
          rowKey={(row) => row.id}
          caption="Calipers"
          empty="No calipers match this filter."
          isRowActive={(row) => row.id === selected || row.id === hovered}
          onRowHover={(row) => onHover(row?.id ?? null)}
          onRowClick={(row) => onSelect(row.id)}
        />
      </div>

      <p className="p-2.5 pt-2 text-[11px] text-fg-subtle">
        Click a caliper, in the list or on the image, to see its profile. <Kbd>↑</Kbd> <Kbd>↓</Kbd>{" "}
        step, <Kbd>F</Kbd> frames, <Kbd>Esc</Kbd> clears.
      </p>
    </Panel>
  );
}

/** The selected caliper's intensity profile, with its edge against the nominal position. */
export function CaliperProfilePanel({ row, mm }: { row: CaliperRow; mm: string | null }) {
  const profile = caliperProfile(row.caliper);
  const edge = row.caliper.profile.edges[0];
  return (
    <Panel
      title={`Caliper ${row.object}.${row.index}`}
      actions={
        <span className={cn("text-[11px]", row.status === "hit" ? "text-signal" : "text-defect")}>
          {row.status === "hit" ? "hit" : reasonText(row.reason)}
        </span>
      }
    >
      <LineProfile
        series={profile.series}
        edges={profile.edges}
        markers={profile.markers}
        {...(profile.xDomain ? { xDomain: profile.xDomain } : {})}
        label={`Intensity along caliper ${row.object}.${row.index}`}
        xLabel="position along the caliper (px)"
        yLabel="intensity"
        variant="fluid"
        height={170}
      />
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[11px]">
        {edge ? (
          <>
            <dt className="text-fg-muted">edge</dt>
            <dd className="font-mono text-fg tabular-nums">
              {edge.pos_px.toFixed(3)} px from nominal, {edge.polarity}
            </dd>
            <dt className="text-fg-muted">residual</dt>
            <dd className="font-mono text-fg tabular-nums">{fixed(row.residual, 3)} px against the fit</dd>
            <dt className="text-fg-muted">amplitude</dt>
            <dd className="font-mono text-fg tabular-nums">{fixed(row.amplitude, 2)}</dd>
            {mm && (
              <>
                <dt className="text-fg-muted">edge (mm)</dt>
                <dd className="font-mono text-fg tabular-nums">{mm}</dd>
              </>
            )}
          </>
        ) : (
          <>
            <dt className="text-fg-muted">verdict</dt>
            <dd className="text-fg">rejected ({reasonText(row.reason)}), so nothing from it went into the fit</dd>
          </>
        )}
      </dl>
    </Panel>
  );
}

function fixed(value: number | null, digits: number): string {
  return value === null ? "—" : value.toFixed(digits);
}
