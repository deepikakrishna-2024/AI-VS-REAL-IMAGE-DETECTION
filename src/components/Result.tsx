import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, RotateCcw } from "lucide-react";

interface ResultProps {
  label: "Real" | "AI Generated";
  confidence: number;
  onReset: () => void;
}

const Result = ({ label, confidence, onReset }: ResultProps) => {
  const isReal = label === "Real";
  const percent = Math.round(confidence * 100);

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", damping: 20 }}
      className={`w-full max-w-lg mx-auto rounded-lg border p-6 ${
        isReal
          ? "border-success/40 bg-success/5 glow-success"
          : "border-destructive/40 bg-destructive/5 glow-destructive"
      }`}
    >
      <div className="flex items-center gap-4 mb-6">
        <div
          className={`rounded-full p-3 ${
            isReal ? "bg-success/20" : "bg-destructive/20"
          }`}
        >
          {isReal ? (
            <ShieldCheck className="h-8 w-8 text-success" />
          ) : (
            <ShieldAlert className="h-8 w-8 text-destructive" />
          )}
        </div>
        <div>
          <p className="text-sm text-muted-foreground font-display uppercase tracking-widest">
            Classification
          </p>
          <h3
            className={`text-2xl font-bold font-display ${
              isReal ? "text-success" : "text-destructive"
            }`}
          >
            {label}
          </h3>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="mb-2">
        <div className="flex justify-between text-sm mb-1.5">
          <span className="text-muted-foreground font-display">Confidence</span>
          <span
            className={`font-bold font-display ${
              isReal ? "text-success" : "text-destructive"
            }`}
          >
            {percent}%
          </span>
        </div>
        <div className="h-3 rounded-full bg-secondary overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${percent}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={`h-full rounded-full ${
              isReal
                ? "bg-gradient-to-r from-success/70 to-success"
                : "bg-gradient-to-r from-destructive/70 to-destructive"
            }`}
          />
        </div>
      </div>

      <button
        onClick={onReset}
        className="mt-6 w-full flex items-center justify-center gap-2 rounded-md border border-border bg-secondary/50 px-4 py-2.5 text-sm font-medium text-foreground hover:bg-secondary transition-colors"
      >
        <RotateCcw className="h-4 w-4" />
        Analyze Another Image
      </button>
    </motion.div>
  );
};

export default Result;
