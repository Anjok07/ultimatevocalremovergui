import { useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TableSortLabel,
  Tooltip,
  Typography,
} from "@mui/material";
import SpeedIcon from "@mui/icons-material/Speed";
import { humanBytes } from "../api";
import type { StatRow, StatsInfo, SystemInfo } from "../types";

type SortKey = "model" | "device" | "runs" | "sec_per_audio_min" | "avg_peak_mb";

// Columns where 0 means "no data" — these always sort to the bottom.
const NO_DATA_ZERO: SortKey[] = ["sec_per_audio_min", "avg_peak_mb"];

function sortValue(r: StatRow, key: SortKey): number | string {
  if (key === "model") return r.model.toLowerCase();
  if (key === "device") return r.device;
  if (key === "runs") return r.completed;
  return r[key];
}

// Per-host processing performance, accumulated over time: how long each model
// takes per minute of audio on this machine, and its peak memory.
export function StatsPanel({ stats, system }: { stats: StatsInfo | null; system: SystemInfo | null }) {
  const allRows = stats?.rows ?? [];
  // The device column is only meaningful when more than one device appears.
  const showDevice = new Set(allRows.map((r) => r.device)).size > 1;

  // Default: fastest first (s/min ascending); rows with no data sink to the bottom.
  const [sortBy, setSortBy] = useState<SortKey>("sec_per_audio_min");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const sortHeader = (key: SortKey) => () => {
    if (sortBy === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortBy(key);
      setSortDir(key === "model" ? "asc" : key === "sec_per_audio_min" ? "asc" : "desc");
    }
  };

  const rows = [...allRows].sort((a, b) => {
    const noData = NO_DATA_ZERO.includes(sortBy);
    const am = noData && Number(a[sortBy as keyof StatRow]) <= 0;
    const bm = noData && Number(b[sortBy as keyof StatRow]) <= 0;
    if (am && bm) return 0;
    if (am) return 1; // missing always last, regardless of direction
    if (bm) return -1;
    const av = sortValue(a, sortBy);
    const bv = sortValue(b, sortBy);
    const r = typeof av === "string" ? av.localeCompare(bv as string) : (av as number) - (bv as number);
    return sortDir === "asc" ? r : -r;
  });

  const sortable = (key: SortKey, label: React.ReactNode) => (
    <TableSortLabel active={sortBy === key} direction={sortBy === key ? sortDir : "asc"} onClick={sortHeader(key)}>
      {label}
    </TableSortLabel>
  );

  const hostLabel = system
    ? system.gpu_available
      ? `${system.name || "GPU"}`
      : system.cpu_model || "CPU"
    : stats?.host_device.toUpperCase() || "…";

  const ramText = system?.ram_total
    ? system.ram_limit && system.ram_limit < system.ram_total
      ? `${humanBytes(system.ram_limit)} cap / ${humanBytes(system.ram_total)}`
      : humanBytes(system.ram_total)
    : null;
  const cpuText = [
    system?.cpu_model,
    system?.cpu_cores ? `${system.cpu_cores} cores` : null,
    system?.cpu_mhz ? `${(system.cpu_mhz / 1000).toFixed(2)} GHz` : null,
  ]
    .filter(Boolean)
    .join(" · ");
  const hostTip = system
    ? [
        cpuText || null,
        ramText ? `${ramText} RAM` : null,
        system.gpu_available ? `GPU: ${system.name}` : "no GPU",
        system.torch ? `torch ${system.torch}` : null,
      ]
        .filter(Boolean)
        .join("\n")
    : "";

  return (
    <Card>
      <CardContent>
        <Stack direction="row" alignItems="center" sx={{ mb: 1.5 }} spacing={1}>
          <SpeedIcon fontSize="small" color="secondary" />
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            Processing performance
          </Typography>
          <Box flexGrow={1} />
          {system && (
            <Tooltip title={<span style={{ whiteSpace: "pre-line" }}>{hostTip}</span>}>
              <Chip
                size="small"
                variant="outlined"
                label={hostLabel}
                sx={{ maxWidth: 220, ".MuiChip-label": { overflow: "hidden", textOverflow: "ellipsis" } }}
              />
            </Tooltip>
          )}
        </Stack>

        {rows.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No completed separations yet. Run a job to start tracking time per minute of audio.
          </Typography>
        ) : (
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small" sx={{ minWidth: 320 }}>
              <TableHead>
                <TableRow>
                  <TableCell>{sortable("model", "Model")}</TableCell>
                  {showDevice && <TableCell align="center">{sortable("device", "Device")}</TableCell>}
                  <TableCell align="right">
                    <Tooltip title="Completed runs (failed/cancelled in red)">
                      <span>{sortable("runs", "Runs")}</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Avg processing seconds per minute of audio on full runs (lower = faster; format-independent)">
                      <span>{sortable("sec_per_audio_min", "s / min")}</span>
                    </Tooltip>
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="Peak memory — average (worst-case)">
                      <span>{sortable("avg_peak_mb", "Peak mem")}</span>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={`${r.model}|${r.device}`}>
                    <TableCell sx={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      <Tooltip title={`${r.model} (${r.arch.toUpperCase()})`}>
                        <span>{r.model}</span>
                      </Tooltip>
                    </TableCell>
                    {showDevice && (
                      <TableCell align="center">
                        <Chip size="small" label={r.device.toUpperCase()} variant="outlined" sx={{ height: 20, fontSize: 11 }} />
                      </TableCell>
                    )}
                    <TableCell align="right">
                      {r.completed}
                      {r.failed + r.cancelled > 0 && (
                        <Tooltip title={`${r.failed} failed, ${r.cancelled} cancelled`}>
                          <Box component="span" sx={{ color: "error.main", ml: 0.5 }}>
                            (+{r.failed + r.cancelled})
                          </Box>
                        </Tooltip>
                      )}
                    </TableCell>
                    <TableCell align="right" sx={{ fontWeight: 600 }}>
                      <Tooltip title={r.total_audio_min > 0 ? `${r.total_audio_min.toFixed(1)} min processed` : ""}>
                        <span>{r.sec_per_audio_min > 0 ? r.sec_per_audio_min.toFixed(1) : "—"}</span>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="right" sx={{ color: "text.secondary" }}>
                      {r.avg_peak_mb > 0 ? `${r.avg_peak_mb.toFixed(0)} (${r.max_peak_mb.toFixed(0)})` : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
