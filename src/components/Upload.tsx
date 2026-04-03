import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload as UploadIcon, X, Image as ImageIcon } from "lucide-react";

interface UploadProps {
  onFileSelect: (file: File) => void;
  isAnalyzing: boolean;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
const MAX_SIZE = 5 * 1024 * 1024; // 5MB

const Upload = ({ onFileSelect, isAnalyzing }: UploadProps) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const validateAndSet = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.includes(file.type)) {
        setError("Only JPG and PNG files are accepted.");
        return;
      }
      if (file.size > MAX_SIZE) {
        setError("File must be under 5 MB.");
        return;
      }
      const url = URL.createObjectURL(file);
      setPreview(url);
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndSet(file);
    },
    [validateAndSet]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndSet(file);
    },
    [validateAndSet]
  );

  const clearPreview = () => {
    if (preview) URL.revokeObjectURL(preview);
    setPreview(null);
    setError(null);
  };

  return (
    <div className="w-full max-w-lg mx-auto">
      <AnimatePresence mode="wait">
        {!preview ? (
          <motion.label
            key="dropzone"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95 }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            className={`relative flex flex-col items-center justify-center gap-4 rounded-lg border-2 border-dashed p-12 cursor-pointer transition-all duration-300 ${
              dragOver
                ? "border-primary bg-primary/5 glow-primary"
                : "border-border hover:border-primary/50 bg-card"
            }`}
          >
            <input
              type="file"
              accept=".jpg,.jpeg,.png"
              onChange={handleFileInput}
              className="hidden"
            />
            <motion.div
              animate={dragOver ? { scale: 1.1 } : { scale: 1 }}
              className="rounded-full bg-secondary p-4"
            >
              <UploadIcon className="h-8 w-8 text-primary" />
            </motion.div>
            <div className="text-center">
              <p className="text-foreground font-medium">
                Drop an image here or{" "}
                <span className="text-primary underline">browse</span>
              </p>
              <p className="text-muted-foreground text-sm mt-1">
                JPG or PNG • Max 5 MB
              </p>
            </div>
          </motion.label>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="relative rounded-lg overflow-hidden border border-border bg-card"
          >
            {!isAnalyzing && (
              <button
                onClick={clearPreview}
                className="absolute top-3 right-3 z-10 rounded-full bg-background/80 p-1.5 hover:bg-destructive/20 transition-colors"
              >
                <X className="h-4 w-4 text-foreground" />
              </button>
            )}
            <div className="relative">
              <img
                src={preview}
                alt="Uploaded preview"
                className="w-full max-h-80 object-contain bg-muted"
              />
              {isAnalyzing && (
                <div className="absolute inset-0 overflow-hidden">
                  <motion.div
                    className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent"
                    animate={{ y: ["-100%", "32000%"] }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  />
                  <div className="absolute inset-0 bg-primary/5" />
                </div>
              )}
            </div>
            <div className="p-3 flex items-center gap-2 text-sm text-muted-foreground">
              <ImageIcon className="h-4 w-4" />
              {isAnalyzing ? (
                <span className="text-primary animate-pulse-glow font-display">
                  Analyzing image...
                </span>
              ) : (
                <span>Image loaded — ready to analyze</span>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {error && (
          <motion.p
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="mt-3 text-sm text-destructive text-center"
          >
            {error}
          </motion.p>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Upload;
