import { useEffect, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  IconButton,
  LinearProgress,
  Stack,
  Tooltip,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import StorageIcon from "@mui/icons-material/Storage";
import ReplayIcon from "@mui/icons-material/Replay";
import StopCircleIcon from "@mui/icons-material/StopCircle";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import VolumeUpIcon from "@mui/icons-material/VolumeUp";
import { api, apiUrl, humanBytes } from "../api";
import type { JobInfo, JobStatus, NowPlaying, StorageInfo } from "../types";

const STATUS_COLOR: Record<JobStatus, "default" | "info" | "success" | "error" | "warning"> = {
  queued: "default",
  running: "info",
  completed: "success",
  failed: "error",
  cancelled: "warning",
};

function fmtSecs(s: number): string {
  if (s >= 60) return `${Math.floor(s / 60)}m${String(Math.round(s % 60)).padStart(2, "0")}s`;
  return `${s.toFixed(s < 10 ? 1 : 0)}s`;
}

function JobCard({
  job,
  onDeleted,
  onReuse,
  onPlay,
  playingUrl,
}: {
  job: JobInfo;
  onDeleted: () => void;
  onReuse: (options: Record<string, unknown>) => void;
  onPlay: (t: NowPlaying) => void;
  playingUrl: string | null;
}) {
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const active = job.status === "running" || job.status === "queued";
  const canReuse = job.kind === "separation" && !!job.options;

  // Live elapsed clock while the job is processing.
  const [nowSec, setNowSec] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNowSec(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [active]);
  const elapsed =
    active && job.started_at ? Math.max(0, nowSec - job.started_at) : null;

  const del = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Delete this job and its files? This cannot be undone."))
      return;
    setDeleting(true);
    try {
      await api.deleteJob(job.id);
      onDeleted();
    } catch (err) {
      console.error("delete failed", err);
      setDeleting(false);
    }
  };

  const cancel = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await api.cancelJob(job.id);
      onDeleted();
    } catch (err) {
      console.error("cancel failed", err);
    }
  };

  const deleteStem = async (filename: string) => {
    if (!window.confirm("Delete this stem file? It can't be undone.")) return;
    try {
      await api.deleteStem(job.id, filename);
      onDeleted();
    } catch (err) {
      console.error("delete stem failed", err);
    }
  };

  return (
    <Accordion
      expanded={open}
      onChange={() => setOpen((o) => !o)}
      disableGutters
      sx={{ bgcolor: "background.default" }}
    >
      <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        sx={{ "& .MuiAccordionSummary-content": { minWidth: 0, overflow: "hidden" } }}
      >
        <Box sx={{ width: "100%", minWidth: 0 }}>
          <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
            <Chip
              size="small"
              label={job.status}
              color={STATUS_COLOR[job.status]}
              variant="outlined"
              sx={{ flexShrink: 0 }}
            />
            <Typography
              variant="body2"
              noWrap
              sx={{ fontWeight: 600, flexGrow: 1, minWidth: 0 }}
            >
              {job.kind === "download"
                ? job.message
                : job.input_filename || job.id}
            </Typography>
            {/* Time + secondary metadata — hidden on phones (also in the details/log) */}
            <Box
              sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center", gap: 1, flexShrink: 0 }}
            >
              {elapsed != null ? (
                <Typography
                  variant="caption"
                  noWrap
                  sx={{ color: "info.main", fontVariantNumeric: "tabular-nums" }}
                >
                  ⏱ {fmtSecs(elapsed)}
                </Typography>
              ) : (
                job.duration_sec != null && (
                  <Tooltip
                    title={
                      job.audio_seconds > 0
                        ? `${job.audio_seconds.toFixed(0)}s audio on ${(job.device || "cpu").toUpperCase()} · ${(
                            (job.duration_sec * 60) / job.audio_seconds
                          ).toFixed(1)} s/min`
                        : ""
                    }
                  >
                    <Typography variant="caption" color="text.secondary" noWrap>
                      {fmtSecs(job.duration_sec)}
                    </Typography>
                  </Tooltip>
                )
              )}
              {job.peak_mem_bytes > 0 && (
                <Tooltip title="Peak memory used by this job">
                  <Typography variant="caption" color="text.secondary" noWrap>
                    {humanBytes(job.peak_mem_bytes)} mem
                  </Typography>
                </Tooltip>
              )}
              {job.bytes > 0 && (
                <Typography variant="caption" color="text.secondary" noWrap>
                  {humanBytes(job.bytes)}
                </Typography>
              )}
              <Typography
                variant="caption"
                color="text.secondary"
                noWrap
                sx={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis" }}
              >
                {(job.options?.model_name as string) || ""}
              </Typography>
            </Box>
            {active && (
              <Tooltip title="Cancel job">
                <IconButton size="small" color="warning" onClick={cancel}>
                  <StopCircleIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            {canReuse && (
              <Tooltip title="Reuse these settings in the form">
                <IconButton
                  size="small"
                  color="primary"
                  onClick={(e) => {
                    e.stopPropagation();
                    onReuse(job.options as Record<string, unknown>);
                  }}
                >
                  <ReplayIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )}
            <Tooltip title="Delete job + files">
              <span>
                <IconButton size="small" color="error" disabled={deleting || active} onClick={del}>
                  {deleting ? <CircularProgress size={16} /> : <DeleteOutlineIcon fontSize="small" />}
                </IconButton>
              </span>
            </Tooltip>
          </Stack>
          {active && (
            <LinearProgress
              variant={job.progress > 0 ? "determinate" : "indeterminate"}
              value={Math.round(job.progress * 100)}
              sx={{ mt: 1 }}
            />
          )}
        </Box>
      </AccordionSummary>
      <AccordionDetails>
        {job.error && (
          <Typography variant="body2" color="error" sx={{ mb: 1 }}>
            {job.error}
          </Typography>
        )}

        {job.outputs.length > 0 && (
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
            {job.outputs.map((o) => {
              if (o.deleted) {
                return (
                  <Chip
                    key={o.filename}
                    size="small"
                    variant="outlined"
                    disabled
                    label={`${o.stem} (deleted)`}
                  />
                );
              }
              const url = apiUrl(`jobs/${job.id}/files/${encodeURIComponent(o.filename)}`);
              const isPlaying = playingUrl === url;
              return (
                <Chip
                  key={o.filename}
                  size="small"
                  color={isPlaying ? "primary" : "default"}
                  variant={isPlaying ? "filled" : "outlined"}
                  icon={isPlaying ? <VolumeUpIcon /> : <PlayArrowIcon />}
                  label={o.stem}
                  onClick={() =>
                    onPlay({
                      url,
                      peaksUrl: apiUrl(`jobs/${job.id}/peaks/${encodeURIComponent(o.filename)}`),
                      label: `${job.input_filename || job.id} — ${o.stem}`,
                      filename: o.filename,
                    })
                  }
                  onDelete={() => deleteStem(o.filename)}
                />
              );
            })}
          </Stack>
        )}

        {job.log && (
          <Box
            component="pre"
            sx={{
              m: 0,
              p: 1.5,
              maxHeight: 200,
              overflow: "auto",
              fontSize: 12,
              // Terminal-style: dark bg + light text so it's readable in both themes.
              bgcolor: "#0b0d12",
              color: "#cbd5e1",
              borderRadius: 1,
              whiteSpace: "pre-wrap",
            }}
          >
            {job.log}
          </Box>
        )}
      </AccordionDetails>
    </Accordion>
  );
}

export function JobsPanel({
  jobs,
  storage,
  onChanged,
  onReuse,
  onPlay,
  playingUrl,
}: {
  jobs: JobInfo[];
  storage: StorageInfo | null;
  onChanged: () => void;
  onReuse: (options: Record<string, unknown>) => void;
  onPlay: (t: NowPlaying) => void;
  playingUrl: string | null;
}) {
  return (
    <Card>
      <CardContent>
        <Stack direction="row" alignItems="center" sx={{ mb: 2 }}>
          <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
            Jobs
          </Typography>
          <Box flexGrow={1} />
          {storage && (
            <Tooltip
              title={`Input ${humanBytes(storage.uploads_bytes)} · Output ${humanBytes(
                storage.outputs_bytes
              )} across ${storage.job_count} job(s)`}
            >
              <Chip
                icon={<StorageIcon />}
                size="small"
                variant="outlined"
                label={`${humanBytes(storage.total_bytes)} on disk`}
              />
            </Tooltip>
          )}
        </Stack>
        {jobs.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No jobs yet. Upload a file and hit Separate.
          </Typography>
        ) : (
          <Stack spacing={1}>
            {jobs.map((j) => (
              <JobCard
                key={j.id}
                job={j}
                onDeleted={onChanged}
                onReuse={onReuse}
                onPlay={onPlay}
                playingUrl={playingUrl}
              />
            ))}
          </Stack>
        )}
      </CardContent>
    </Card>
  );
}
