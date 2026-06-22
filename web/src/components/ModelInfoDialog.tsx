import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Box,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableRow,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { humanBytes } from "../api";
import type { ModelDetail } from "../types";

export function ModelInfoDialog({
  open,
  onClose,
  detail,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  detail: ModelDetail | null;
  loading: boolean;
}) {
  const tech = detail?.technical ?? {};
  const techKeys = Object.keys(tech);

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="xs">
      <DialogTitle sx={{ pb: 0 }}>
        {detail?.name || "Model"}
        <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
          {detail?.download_name}
        </Typography>
      </DialogTitle>
      <DialogContent>
        {loading || !detail ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 3 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <Stack spacing={1.5} sx={{ mt: 1 }}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip size="small" label={detail.arch.toUpperCase()} color="primary" variant="outlined" />
              <Chip
                size="small"
                label={detail.installed ? "installed" : "not installed"}
                color={detail.installed ? "success" : "default"}
                variant="outlined"
              />
              {detail.bytes > 0 && <Chip size="small" variant="outlined" label={humanBytes(detail.bytes)} />}
            </Stack>

            {detail.stems.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary">
                  Produces stems
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 0.5 }}>
                  {detail.stems.map((s) => (
                    <Chip key={s} size="small" label={s} color="secondary" />
                  ))}
                </Stack>
              </Box>
            )}

            {detail.note && (
              <Typography variant="body2" color="text.secondary">
                {detail.note}
              </Typography>
            )}

            {techKeys.length > 0 && (
              <Accordion disableGutters elevation={0} sx={{ bgcolor: "transparent", "&:before": { display: "none" } }}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ px: 0 }}>
                  <Typography variant="body2" color="text.secondary">
                    Technical details
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ px: 0, pt: 0 }}>
                  <Table size="small">
                    <TableBody>
                      {techKeys.map((k) => (
                        <TableRow key={k}>
                          <TableCell sx={{ color: "text.secondary", border: 0, py: 0.5 }}>{k}</TableCell>
                          <TableCell align="right" sx={{ border: 0, py: 0.5, fontFamily: "monospace" }}>
                            {String(tech[k])}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </AccordionDetails>
              </Accordion>
            )}
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  );
}
