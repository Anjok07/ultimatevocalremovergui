import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AppBar,
  Badge,
  Box,
  Chip,
  Container,
  CssBaseline,
  Drawer,
  Grid,
  IconButton,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
} from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";
import MemoryIcon from "@mui/icons-material/Memory";
import BoltIcon from "@mui/icons-material/Bolt";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import LightModeIcon from "@mui/icons-material/LightMode";
import QueueMusicIcon from "@mui/icons-material/QueueMusic";
import CloseIcon from "@mui/icons-material/Close";
import { api } from "./api";
import type {
  HelpTexts,
  InputInfo,
  JobInfo,
  ModelInfo,
  NowPlaying,
  StatsInfo,
  StorageInfo,
  SystemInfo,
} from "./types";
import { SeparationForm } from "./components/SeparationForm";
import { JobsPanel } from "./components/JobsPanel";
import { StatsPanel } from "./components/StatsPanel";
import { MiniPlayer } from "./components/MiniPlayer";
import { makeTheme, type Mode } from "./theme";
import { usePersisted } from "./usePersisted";

export function App() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [jobs, setJobs] = useState<JobInfo[]>([]);
  const [inputs, setInputs] = useState<InputInfo[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [stats, setStats] = useState<StatsInfo | null>(null);
  const [help, setHelp] = useState<HelpTexts>({});
  const [preset, setPreset] = useState<Record<string, unknown> | null>(null);
  const [nowPlaying, setNowPlaying] = useState<NowPlaying | null>(null);

  // Theme: default to the system setting; persist an explicit choice.
  const prefersDark = useMediaQuery("(prefers-color-scheme: dark)");
  const [modePref, setModePref] = usePersisted<Mode | "system">("uvr.themeMode", "system");
  const mode: Mode = modePref === "system" ? (prefersDark ? "dark" : "light") : modePref;
  const theme = useMemo(() => makeTheme(mode), [mode]);
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const [drawerOpen, setDrawerOpen] = useState(false);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await api.listModels());
    } catch (e) {
      console.error("Failed to load models", e);
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await api.listJobs());
      setStorage(await api.getStorage());
      setStats(await api.getStats());
    } catch (e) {
      console.error("Failed to load jobs", e);
    }
  }, []);

  const refreshInputs = useCallback(async () => {
    try {
      setInputs(await api.listInputs());
    } catch (e) {
      console.error("Failed to load inputs", e);
    }
  }, []);

  useEffect(() => {
    refreshModels();
    refreshJobs();
    refreshInputs();
    api.getSystem().then(setSystem).catch((e) => console.error("system info", e));
    api.getHelp().then(setHelp).catch((e) => console.error("help", e));
    const t = setInterval(refreshJobs, 2000);
    return () => clearInterval(t);
  }, [refreshModels, refreshJobs, refreshInputs]);

  const busyJob = jobs.find(
    (j) => j.kind === "separation" && (j.status === "running" || j.status === "queued")
  );

  const gpu = system?.gpu_available ?? false;
  const deviceLabel = system ? (gpu ? `GPU · ${system.name || system.device.toUpperCase()}` : "CPU") : "…";
  const deviceTip = system
    ? `Separations run on ${system.device.toUpperCase()}` +
      (system.torch ? ` · torch ${system.torch}` : "") +
      ` · mode: ${system.mode}`
    : "Detecting compute device…";

  const jobsAndStats = (
    <Stack spacing={3}>
      <JobsPanel
        jobs={jobs}
        storage={storage}
        onChanged={refreshJobs}
        onReuse={setPreset}
        onPlay={setNowPlaying}
        playingUrl={nowPlaying?.url ?? null}
      />
      <StatsPanel stats={stats} system={system} />
    </Stack>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: "100vh" }}>
        <AppBar position="static" color="transparent" elevation={0}>
          <Toolbar>
            <GraphicEqIcon sx={{ mr: 1.5, color: "primary.main" }} />
            <Typography variant="h6" sx={{ fontWeight: 700 }}>
              Ultimate Vocal Remover
            </Typography>
            <Typography variant="body2" sx={{ ml: 1.5, color: "text.secondary", display: { xs: "none", sm: "block" } }}>
              web service
            </Typography>
            <Box flexGrow={1} />
            <Tooltip title={deviceTip}>
              <Chip
                icon={gpu ? <BoltIcon /> : <MemoryIcon />}
                label={deviceLabel}
                color={gpu ? "secondary" : "default"}
                variant={gpu ? "filled" : "outlined"}
                size="small"
                sx={{ mr: 1 }}
              />
            </Tooltip>
            <Tooltip title={mode === "dark" ? "Switch to light" : "Switch to dark"}>
              <IconButton onClick={() => setModePref(mode === "dark" ? "light" : "dark")} color="inherit">
                {mode === "dark" ? <LightModeIcon /> : <DarkModeIcon />}
              </IconButton>
            </Tooltip>
            {isMobile && (
              <Tooltip title="Jobs & performance">
                <IconButton color="inherit" onClick={() => setDrawerOpen(true)}>
                  <Badge badgeContent={jobs.length} color="primary" max={99}>
                    <QueueMusicIcon />
                  </Badge>
                </IconButton>
              </Tooltip>
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ py: { xs: 2, md: 4 }, px: { xs: 1.5, md: 3 } }}>
          {isMobile ? (
            <Stack spacing={3}>
              <SeparationForm
                models={models}
                inputs={inputs}
                help={help}
                preset={preset}
                busy={!!busyJob}
                onPresetApplied={() => setPreset(null)}
                onModelsChanged={refreshModels}
                onInputsChanged={refreshInputs}
                onJobCreated={() => {
                  refreshJobs();
                  refreshInputs();
                }}
              />
            </Stack>
          ) : (
            <Grid container spacing={3}>
              <Grid item xs={12} md={5}>
                <Stack spacing={3}>
                  <SeparationForm
                    models={models}
                    inputs={inputs}
                    help={help}
                    preset={preset}
                    busy={!!busyJob}
                    onPresetApplied={() => setPreset(null)}
                    onModelsChanged={refreshModels}
                    onInputsChanged={refreshInputs}
                    onJobCreated={() => {
                      refreshJobs();
                      refreshInputs();
                    }}
                  />
                  <StatsPanel stats={stats} system={system} />
                </Stack>
              </Grid>
              <Grid item xs={12} md={7}>
                <JobsPanel
                  jobs={jobs}
                  storage={storage}
                  onChanged={refreshJobs}
                  onReuse={setPreset}
                  onPlay={setNowPlaying}
                  playingUrl={nowPlaying?.url ?? null}
                />
              </Grid>
            </Grid>
          )}
        </Container>

        {/* Mobile: jobs + performance in a slide-up drawer */}
        <Drawer
          anchor="bottom"
          open={isMobile && drawerOpen}
          onClose={() => setDrawerOpen(false)}
          PaperProps={{ sx: { maxHeight: "85vh", borderTopLeftRadius: 16, borderTopRightRadius: 16 } }}
        >
          <Box sx={{ p: 2 }}>
            <Stack direction="row" alignItems="center" sx={{ mb: 1 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                Jobs & performance
              </Typography>
              <Box flexGrow={1} />
              <IconButton onClick={() => setDrawerOpen(false)}>
                <CloseIcon />
              </IconButton>
            </Stack>
            {jobsAndStats}
          </Box>
        </Drawer>

        {/* Spacer so the fixed player never covers content */}
        {nowPlaying && <Box sx={{ height: 72 }} />}
        <MiniPlayer track={nowPlaying} onClose={() => setNowPlaying(null)} />
      </Box>
    </ThemeProvider>
  );
}
