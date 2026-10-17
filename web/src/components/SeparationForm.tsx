import { useEffect, useMemo, useState } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import DownloadIcon from "@mui/icons-material/Download";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import TuneIcon from "@mui/icons-material/Tune";
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import { api, humanBytes } from "../api";
import type { Arch, HelpTexts, InputInfo, ModelDetail, ModelInfo } from "../types";
import { usePersisted } from "../usePersisted";
import { HelpTip } from "./HelpTip";
import { ModelInfoDialog } from "./ModelInfoDialog";

const ARCHS: { value: Arch; label: string }[] = [
  { value: "mdx", label: "MDX-Net" },
  { value: "vr", label: "VR Arch" },
  { value: "demucs", label: "Demucs" },
];

const FORMATS = ["WAV", "FLAC", "MP3"];
const WINDOW_SIZES = [320, 512, 1024];
const NEW_INPUT = "__new__";

interface Props {
  models: ModelInfo[];
  inputs: InputInfo[];
  help: HelpTexts;
  preset: Record<string, unknown> | null;
  busy: boolean;
  onPresetApplied: () => void;
  onModelsChanged: () => void;
  onInputsChanged: () => void;
  onJobCreated: () => void;
}

export function SeparationForm({
  models,
  inputs,
  help,
  preset,
  busy,
  onPresetApplied,
  onModelsChanged,
  onInputsChanged,
  onJobCreated,
}: Props) {
  // Persisted form state — survives reloads.
  const [arch, setArch] = usePersisted<Arch>("uvr.arch", "mdx");
  const [modelName, setModelName] = usePersisted("uvr.model", "");
  const [format, setFormat] = usePersisted("uvr.format", "WAV");
  const [primaryOnly, setPrimaryOnly] = usePersisted("uvr.primaryOnly", false);
  const [secondaryOnly, setSecondaryOnly] = usePersisted("uvr.secondaryOnly", false);
  const [normalization, setNormalization] = usePersisted("uvr.normalize", false);
  const [denoise, setDenoise] = usePersisted("uvr.denoise", false);
  const [pitchShift, setPitchShift] = usePersisted("uvr.pitch", "0");
  const [overlap, setOverlap] = usePersisted("uvr.overlap", "");
  const [segmentSize, setSegmentSize] = usePersisted("uvr.segment", "256");
  const [aggression, setAggression] = usePersisted("uvr.aggression", "10");
  const [windowSize, setWindowSize] = usePersisted("uvr.window", "512");
  const [tta, setTta] = usePersisted("uvr.tta", false);
  const [postProcess, setPostProcess] = usePersisted("uvr.postProcess", false);
  const [highEnd, setHighEnd] = usePersisted("uvr.highEnd", false);
  const [shifts, setShifts] = usePersisted("uvr.shifts", "2");
  const [demucsSegment, setDemucsSegment] = usePersisted("uvr.demucsSegment", "");
  const [sampleMode, setSampleMode] = usePersisted("uvr.sampleMode", false);
  const [sampleSeconds, setSampleSeconds] = usePersisted("uvr.sampleSeconds", 15);

  // Ephemeral: input choice + upload + model-info dialog.
  const [inputId, setInputId] = useState<string>(NEW_INPUT);
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [infoOpen, setInfoOpen] = useState(false);
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const archModels = useMemo(() => models.filter((m) => m.arch === arch), [models, arch]);
  const selected = archModels.find((m) => m.name === modelName);
  const usingNew = inputId === NEW_INPUT;

  // Apply settings lifted from a past job (one-shot).
  useEffect(() => {
    if (!preset) return;
    const s = (k: string, fb = "") =>
      preset[k] !== undefined && preset[k] !== null ? String(preset[k]) : fb;
    const b = (k: string) => preset[k] === true || preset[k] === "true";
    if (preset.arch) setArch(preset.arch as Arch);
    if (preset.model_name) setModelName(String(preset.model_name));
    if (preset.output_format) setFormat(String(preset.output_format));
    setPrimaryOnly(b("primary_stem_only"));
    setSecondaryOnly(b("secondary_stem_only"));
    setNormalization(b("normalization"));
    setDenoise(b("denoise"));
    setPitchShift(s("semitone_shift", "0"));
    setSegmentSize(s("segment_size", "256"));
    setOverlap(preset.overlap != null ? String(preset.overlap) : "");
    setAggression(s("aggression", "10"));
    setWindowSize(s("window_size", "512"));
    setTta(b("tta"));
    setPostProcess(b("post_process"));
    setHighEnd(b("high_end_process"));
    setShifts(s("shifts", "2"));
    setDemucsSegment(preset.demucs_segment != null ? String(preset.demucs_segment) : "");
    setError(null);
    onPresetApplied();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset]);

  const showModelInfo = async () => {
    if (!modelName) return;
    setInfoOpen(true);
    setDetailLoading(true);
    try {
      setDetail(await api.getModelDetail(arch, modelName));
    } catch (e) {
      console.error(e);
    } finally {
      setDetailLoading(false);
    }
  };

  const submit = async () => {
    setError(null);
    if (usingNew && !file) return setError("Choose an audio file to upload.");
    if (!usingNew && !inputs.some((i) => i.id === inputId)) return setError("Select an input.");
    if (!modelName) return setError("Choose a model.");
    if (!selected?.installed) return setError("Model is not installed — download it first.");

    setSubmitting(true);
    try {
      const form = new FormData();
      if (usingNew && file) form.append("file", file);
      else form.append("input_id", inputId);
      form.append("arch", arch);
      form.append("model_name", modelName);
      form.append("output_format", format);
      form.append("primary_stem_only", String(primaryOnly));
      form.append("secondary_stem_only", String(secondaryOnly));
      form.append("normalization", String(normalization));
      form.append("denoise", String(denoise));
      form.append("semitone_shift", pitchShift || "0");
      form.append("sample_mode", String(sampleMode));
      form.append("sample_seconds", String(sampleSeconds));
      if (arch === "mdx") {
        form.append("segment_size", segmentSize || "256");
        if (overlap !== "") form.append("overlap", overlap);
      } else if (arch === "vr") {
        form.append("aggression", aggression || "10");
        form.append("window_size", windowSize);
        form.append("tta", String(tta));
        form.append("post_process", String(postProcess));
        form.append("high_end_process", String(highEnd));
      } else if (arch === "demucs") {
        form.append("shifts", shifts || "2");
        if (overlap !== "") form.append("overlap", overlap);
        if (demucsSegment !== "") form.append("demucs_segment", demucsSegment);
      }
      await api.separate(form);
      onJobCreated();
      if (usingNew) {
        setFile(null);
        onInputsChanged();
      }
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setSubmitting(false);
    }
  };

  const download = async () => {
    if (!selected) return;
    setDownloading(true);
    setError(null);
    try {
      await api.downloadModel(arch, selected.download_name);
      const started = Date.now();
      const t = setInterval(async () => {
        onModelsChanged();
        if (Date.now() - started > 1000 * 60 * 10) clearInterval(t);
      }, 3000);
    } catch (e) {
      setError(String(e instanceof Error ? e.message : e));
    } finally {
      setDownloading(false);
    }
  };

  const deleteInput = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm("Delete this input file? Existing job outputs are kept.")) return;
    try {
      await api.deleteInput(id);
      if (inputId === id) setInputId(NEW_INPUT);
      onInputsChanged();
    } catch (err) {
      setError(String(err instanceof Error ? err.message : err));
    }
  };

  // Numeric field with an inline help icon.
  const num = (label: string, value: string, setter: (v: string) => void, helpKey: string, props = {}) => (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
      <TextField
        label={label}
        type="number"
        size="small"
        value={value}
        onChange={(e) => setter(e.target.value)}
        fullWidth
        {...props}
      />
      <HelpTip text={help[helpKey]} />
    </Box>
  );

  const toggle = (label: string, checked: boolean, setter: (v: boolean) => void, helpKey: string) => (
    <Box sx={{ display: "flex", alignItems: "center" }}>
      <FormControlLabel
        control={<Switch checked={checked} onChange={(e) => setter(e.target.checked)} />}
        label={label}
      />
      <HelpTip text={help[helpKey]} />
    </Box>
  );

  return (
    <Card>
      <CardContent>
        <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
          New separation
        </Typography>

        <Stack spacing={2}>
          <FormControl fullWidth size="small">
            <InputLabel>Architecture</InputLabel>
            <Select
              label="Architecture"
              value={arch}
              onChange={(e) => {
                setArch(e.target.value as Arch);
                setModelName("");
              }}
            >
              {ARCHS.map((a) => (
                <MenuItem key={a.value} value={a.value}>
                  {a.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box sx={{ display: "flex", gap: 0.5, alignItems: "center" }}>
            <FormControl fullWidth size="small">
              <InputLabel>Model</InputLabel>
              <Select label="Model" value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {archModels.map((m) => (
                  <MenuItem key={m.name} value={m.name}>
                    {m.installed ? "● " : "○ "}
                    {m.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <Tooltip title="Model info">
              <span>
                <IconButton color="info" disabled={!modelName} onClick={showModelInfo}>
                  <InfoOutlinedIcon />
                </IconButton>
              </span>
            </Tooltip>
            <Tooltip title={selected?.installed ? "Installed" : "Download model"}>
              <span>
                <IconButton
                  color="secondary"
                  disabled={!modelName || selected?.installed || downloading}
                  onClick={download}
                >
                  {downloading ? <CircularProgress size={20} /> : <DownloadIcon />}
                </IconButton>
              </span>
            </Tooltip>
          </Box>

          {/* Input source: reuse a saved input or upload a new one */}
          <FormControl fullWidth size="small">
            <InputLabel>Input</InputLabel>
            <Select
              label="Input"
              value={inputId}
              onChange={(e) => setInputId(e.target.value)}
              renderValue={(val) => {
                if (val === NEW_INPUT) return "⬆ Upload new file…";
                const inp = inputs.find((i) => i.id === val);
                return inp ? `${inp.filename} (${humanBytes(inp.bytes)})` : "Select input";
              }}
            >
              <MenuItem value={NEW_INPUT}>⬆ Upload new file…</MenuItem>
              {inputs.map((i) => (
                <MenuItem key={i.id} value={i.id}>
                  <Box sx={{ display: "flex", alignItems: "center", width: "100%", gap: 1 }}>
                    <Box sx={{ flexGrow: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {i.filename}
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {humanBytes(i.bytes)}
                    </Typography>
                    <IconButton size="small" color="error" onClick={(e) => deleteInput(i.id, e)}>
                      <DeleteOutlineIcon fontSize="small" />
                    </IconButton>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {usingNew && (
            <Button component="label" variant="outlined" startIcon={<UploadFileIcon />} sx={{ justifyContent: "flex-start" }}>
              {file ? file.name : "Choose audio file"}
              <input
                hidden
                type="file"
                accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </Button>
          )}

          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
            <FormControl fullWidth size="small">
              <InputLabel>Output format</InputLabel>
              <Select label="Output format" value={format} onChange={(e) => setFormat(e.target.value)}>
                {FORMATS.map((f) => (
                  <MenuItem key={f} value={f}>
                    {f}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <HelpTip text={help.output_format} />
          </Box>

          <Divider flexItem />

          <Stack>
            {toggle("Primary stem only", primaryOnly, setPrimaryOnly, "primary_stem_only")}
            {toggle("Secondary stem only", secondaryOnly, setSecondaryOnly, "secondary_stem_only")}
            {toggle("Normalize output", normalization, setNormalization, "normalization")}
            {toggle("Denoise", denoise, setDenoise, "denoise")}
            {toggle(`Sample mode${sampleMode ? ` (${sampleSeconds}s)` : ""}`, sampleMode, setSampleMode, "sample_mode")}
          </Stack>

          {sampleMode && (
            <Box sx={{ px: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Process only a {sampleSeconds}s clip (quick preview)
              </Typography>
              <Slider
                value={sampleSeconds}
                onChange={(_, v) => setSampleSeconds(v as number)}
                min={5}
                max={30}
                step={1}
                marks={[
                  { value: 5, label: "5s" },
                  { value: 15, label: "15s" },
                  { value: 30, label: "30s" },
                ]}
                valueLabelDisplay="auto"
                size="small"
              />
            </Box>
          )}

          {/* Arch-aware advanced parameters */}
          <Accordion disableGutters elevation={0} sx={{ bgcolor: "transparent", "&:before": { display: "none" } }}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
              <TuneIcon fontSize="small" sx={{ mr: 1, color: "text.secondary" }} />
              <Typography variant="body2" sx={{ color: "text.secondary" }}>
                Advanced ({ARCHS.find((a) => a.value === arch)?.label} settings)
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ px: 0 }}>
              <Stack spacing={2}>
                {arch === "mdx" && (
                  <>
                    {num("Segment size", segmentSize, setSegmentSize, "segment_size", { inputProps: { min: 32, step: 32 } })}
                    {num("Overlap (blank = default)", overlap, setOverlap, "overlap", {
                      inputProps: { min: 0, max: 0.99, step: 0.05 },
                      placeholder: "Default",
                    })}
                  </>
                )}
                {arch === "vr" && (
                  <>
                    {num("Aggression (0–100)", aggression, setAggression, "aggression", { inputProps: { min: 0, max: 100 } })}
                    <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                      <FormControl fullWidth size="small">
                        <InputLabel>Window size</InputLabel>
                        <Select label="Window size" value={windowSize} onChange={(e) => setWindowSize(e.target.value)}>
                          {WINDOW_SIZES.map((w) => (
                            <MenuItem key={w} value={String(w)}>
                              {w}
                            </MenuItem>
                          ))}
                        </Select>
                      </FormControl>
                      <HelpTip text={help.window_size} />
                    </Box>
                    {toggle("TTA (test-time augmentation)", tta, setTta, "tta")}
                    {toggle("Post-process", postProcess, setPostProcess, "post_process")}
                    {toggle("High-end process", highEnd, setHighEnd, "high_end_process")}
                  </>
                )}
                {arch === "demucs" && (
                  <>
                    {num("Shifts", shifts, setShifts, "shifts", { inputProps: { min: 0, max: 10 } })}
                    {num("Overlap (blank = default)", overlap, setOverlap, "overlap", {
                      inputProps: { min: 0, max: 0.99, step: 0.05 },
                      placeholder: "Default",
                    })}
                    {num("Segment (blank = default)", demucsSegment, setDemucsSegment, "demucs_segment", {
                      inputProps: { min: 1 },
                      placeholder: "Default",
                    })}
                  </>
                )}
                {num("Pitch shift (semitones)", pitchShift, setPitchShift, "pitch_shift", { inputProps: { step: 1 } })}

                <Typography variant="caption" color="text.secondary">
                  Compute device is auto-detected (GPU when available, else CPU) and shown in the header —
                  there's no manual toggle, since you'd always want the GPU when one is present.
                </Typography>
              </Stack>
            </AccordionDetails>
          </Accordion>

          {error && <Alert severity="error">{error}</Alert>}
          {busy && (
            <Alert severity="info" sx={{ py: 0 }}>
              A separation is running — only one job runs at a time.
            </Alert>
          )}

          <Button
            variant="contained"
            size="large"
            disabled={submitting || busy}
            onClick={submit}
            startIcon={submitting || busy ? <CircularProgress size={18} /> : undefined}
          >
            {busy ? "Job running…" : submitting ? "Submitting…" : "Separate"}
          </Button>
        </Stack>
      </CardContent>

      <ModelInfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} detail={detail} loading={detailLoading} />
    </Card>
  );
}
