/**
 * Auto-start script for AI vs Real Image Detection
 * - Checks/installs Python requirements (if needed)
 * - Trains model (if not exists)
 * - Starts ML API backend
 * - Starts Vite frontend
 */

const { spawn, exec } = require('child_process');
const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..');
const ML_MODEL_DIR = path.join(PROJECT_ROOT, 'ml_model');
const API_DIR = path.join(PROJECT_ROOT, 'api');
const MODEL_FILE = path.join(ML_MODEL_DIR, 'ai_real_classifier.pth');
const CLASS_INDICES_FILE = path.join(ML_MODEL_DIR, 'class_indices.pkl');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}[AutoStart] ${message}${colors.reset}`);
}

function runCommand(command, cwd, description) {
  return new Promise((resolve, reject) => {
    log(`${description}...`, 'cyan');
    
    const isWindows = process.platform === 'win32';
    const shell = isWindows ? 'cmd.exe' : 'sh';
    const shellFlag = isWindows ? '/c' : '-c';
    
    const child = spawn(shell, [shellFlag, command], {
      cwd: cwd,
      stdio: 'pipe'
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
      process.stdout.write(data);
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
      process.stderr.write(data);
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Command failed with code ${code}: ${stderr}`));
      }
    });
  });
}

function checkPythonPackage(packageName) {
  return new Promise((resolve) => {
    exec(`python -c "import ${packageName}"`, (error) => {
      resolve(!error);
    });
  });
}

async function checkAndInstallRequirements() {
  // Skip package checking - it's slow and we assume packages are installed
  // Users should manually install if needed: pip install -r requirements.txt
  log('Skipping Python package check (assumes installed)', 'blue');
}

async function checkAndTrainModel() {
  // Check if model files exist
  const modelExists = fs.existsSync(MODEL_FILE);
  const classIndicesExist = fs.existsSync(CLASS_INDICES_FILE);
  
  if (modelExists && classIndicesExist) {
    log('Model files found ✓', 'green');
    return;
  }
  
  if (!modelExists) {
    log('Model file not found: ai_real_classifier.pth', 'yellow');
  }
  if (!classIndicesExist) {
    log('Class indices not found: class_indices.pkl', 'yellow');
  }
  
  log('Starting model training...', 'yellow');
  console.log(); // Empty line before training output
  
  try {
    // Run training with inherited stdio to show formatted output
    const isWindows = process.platform === 'win32';
    const pythonCmd = isWindows ? 'python' : 'python3';
    
    await new Promise((resolve, reject) => {
      const trainProcess = spawn(pythonCmd, ['train.py'], {
        cwd: ML_MODEL_DIR,
        stdio: 'inherit' // This shows the output directly
      });
      
      trainProcess.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Training exited with code ${code}`));
        }
      });
      
      trainProcess.on('error', (err) => {
        reject(err);
      });
    });
    
    console.log(); // Empty line after training
    log('Model training complete ✓', 'green');
  } catch (error) {
    log('Model training failed', 'red');
    throw error;
  }
}

function startApiServer() {
  log('Starting ML API server...', 'cyan');
  
  const isWindows = process.platform === 'win32';
  const command = isWindows ? 'python main.py' : 'python main.py';
  
  const apiProcess = spawn('python', ['main.py'], {
    cwd: API_DIR,
    stdio: 'pipe',
    detached: !isWindows // Allow it to run independently
  });

  apiProcess.stdout.on('data', (data) => {
    const output = data.toString();
    if (output.includes('Model loaded') || output.includes('Uvicorn running')) {
      log('ML API server started on http://localhost:8000 ✓', 'green');
    }
    process.stdout.write(`[API] ${output}`);
  });

  apiProcess.stderr.on('data', (data) => {
    process.stderr.write(`[API Error] ${data}`);
  });

  apiProcess.on('close', (code) => {
    if (code !== 0 && code !== null) {
      log(`API server exited with code ${code}`, 'red');
    }
  });

  // Give the API time to start
  return new Promise((resolve) => {
    setTimeout(() => {
      log('API server should be ready', 'green');
      resolve(apiProcess);
    }, 1000);
  });
}

async function main() {
  console.log('\n');
  log('='.repeat(60), 'cyan');
  log('AI vs Real Image Detection - Auto Start', 'cyan');
  log('='.repeat(60), 'cyan');
  console.log('\n');

  try {
    // Step 1: Check and install requirements
    await checkAndInstallRequirements();
    console.log('\n');

    // Step 2: Check and train model
    await checkAndTrainModel();
    console.log('\n');

    // Step 3: Start API server
    const apiProcess = await startApiServer();
    console.log('\n');

    // Step 4: Start Vite frontend (this will take over the console)
    log('Starting Vite frontend...', 'cyan');
    log('Frontend will be available at http://localhost:8080', 'blue');
    log('Press Ctrl+C to stop all services', 'yellow');
    console.log('\n');

    // Get npm path
    const nodeDir = process.env.NODE_DIR || 'C:\\Program Files\\nodejs';
    
    // Use full command string with shell to avoid deprecation warning
    const npmCmd = process.platform === 'win32' 
      ? `"${nodeDir}\\npm.cmd" run dev:raw`
      : 'npm run dev:raw';
    
    const viteProcess = spawn(npmCmd, {
      cwd: PROJECT_ROOT,
      stdio: 'inherit',
      shell: true,
      windowsVerbatimArguments: process.platform === 'win32'
    });

    // Handle cleanup
    process.on('SIGINT', () => {
      log('\nShutting down services...', 'yellow');
      if (apiProcess) apiProcess.kill();
      viteProcess.kill();
      process.exit(0);
    });

    process.on('SIGTERM', () => {
      if (apiProcess) apiProcess.kill();
      viteProcess.kill();
    });

  } catch (error) {
    log(`Error: ${error.message}`, 'red');
    process.exit(1);
  }
}

main();
