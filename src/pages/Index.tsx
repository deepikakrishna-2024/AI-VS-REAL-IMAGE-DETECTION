import { useState, useCallback, useEffect } from "react";
import Upload from "@/components/Upload";
import Result from "@/components/Result";
import HowItWorks from "@/components/HowItWorks";

type PredictionResult = {
  label: "Real" | "AI Generated";
  confidence: number;
};

const API_URL = "http://localhost:8000";

const checkApiHealth = async (): Promise<boolean> => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);
    
    const response = await fetch(`${API_URL}/health`, { 
      method: 'GET',
      signal: controller.signal
    });
    clearTimeout(timeoutId);
    return response.ok;
  } catch {
    return false;
  }
};

const predictImage = async (file: File): Promise<PredictionResult> => {
  const formData = new FormData();
  formData.append("file", file);

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    
    return {
      label: data.label as "Real" | "AI Generated",
      confidence: data.confidence,
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof Error) {
      if (error.name === 'AbortError') {
        throw new Error("Request timed out. The model may be slow on your CPU.");
      }
      if (error.message.includes('fetch') || error.message.includes('Failed to fetch')) {
        throw new Error("Cannot connect to API. Please ensure the ML backend is running on port 8000.");
      }
    }
    throw error;
  }
};

const Index = () => {
  const [file, setFile] = useState<File | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Silent API check in console only
  useEffect(() => {
    checkApiHealth().then(connected => {
      console.log('[API]', connected ? 'Connected' : 'Not connected');
    });
  }, []);

  const handleFileSelect = useCallback((f: File) => {
    setFile(f);
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!file) return;
    
    setAnalyzing(true);
    setError(null);
    
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setAnalyzing(false);
    }
  }, [file]);

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
  };

  return (
    <div style={{ backgroundColor: '#0f172a', minHeight: '100vh', color: '#ffffff', padding: '20px' }}>
      <div style={{ maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
        <h1 style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '10px' }}>
          AI vs Real
        </h1>
        <p style={{ color: '#94a3b8', marginBottom: '20px' }}>
          Upload any image and our neural network will determine whether it's AI-generated or real.
        </p>
        
        <Upload onFileSelect={handleFileSelect} isAnalyzing={analyzing} />

        {file && !result && !analyzing && (
          <button
            onClick={handleAnalyze}
            style={{
              marginTop: '20px',
              padding: '12px 32px',
              backgroundColor: '#3b82f6',
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '1rem',
              fontWeight: '600',
              cursor: 'pointer'
            }}
          >
            Analyze Image
          </button>
        )}

        {error && (
          <div style={{ 
            marginTop: '20px', 
            padding: '12px 20px', 
            backgroundColor: 'rgba(239, 68, 68, 0.1)', 
            border: '1px solid rgba(239, 68, 68, 0.3)',
            borderRadius: '8px',
            color: '#ef4444'
          }}>
            {error}
          </div>
        )}

        {result && (
          <div style={{ marginTop: '20px' }}>
            <Result
              label={result.label}
              confidence={result.confidence}
              onReset={handleReset}
            />
          </div>
        )}

        <HowItWorks />
      </div>
    </div>
  );
};

export default Index;
