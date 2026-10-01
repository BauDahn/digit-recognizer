import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI, Type } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = process.env.PORT || 3000;
const isProd = process.env.NODE_ENV === 'production';

// Allow large image uploads from canvas (up to 25MB)
app.use(express.json({ limit: '25mb' }));

// Server-side Gemini initialization with required user-agent header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

// API endpoint for Handwritten Math, Equation, and Symbol Recognition & Solving
app.post('/api/math/recognize', async (req, res) => {
  try {
    const { imageBase64, mode = 'recognize' } = req.body;

    if (!imageBase64) {
      return res.status(400).json({ error: 'Missing imageBase64 data' });
    }

    // Strip data URL header if present (e.g. "data:image/png;base64,")
    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, '');

    const promptText = `
You are an expert handwriting recognizer and mathematical engine modeled after Microsoft OneNote Math Assistant and Apple Math Notes.
Analyze this handwritten canvas containing mathematical formulas, equations, alphanumeric characters, scientific symbols, or geometric expressions.

Tasks:
1. Recognize the exact handwriting and convert it to clean, standard LaTeX.
2. Provide a plain-text human-readable format.
3. List all distinct symbols/characters identified (digits 0-9, latin variables a-z/A-Z, greek letters \\alpha, \\beta, \\theta, \\pi, operators +, -, \\times, \\div, \\int, \\sum, \\sqrt, =, etc.).
4. Determine if it is a solvable math problem, equation, or arithmetic evaluation.
5. If solvable or if it contains an arithmetic expression (especially if it ends with '=' or is an equation like '3x + 6 = 24'), calculate the solution and provide concise step-by-step reasoning.
6. Provide 'autoAnswer' which is the concise string result that can be written immediately after the '=' sign on a digital notebook (e.g., if handwriting is '14 * 3 =', autoAnswer is '42'; if 'x^2 = 16', autoAnswer is 'x = ±4').
7. If it represents a function of x (like 'y = x^2 - 4' or 'f(x) = sin(x)'), specify 'isGraphable': true, provide a valid JavaScript expression for f(x) (e.g. 'Math.sin(x)', 'Math.pow(x, 2) - 4') and default x domain [-10, 10].
`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.8-flash',
      contents: {
        parts: [
          {
            inlineData: {
              mimeType: 'image/png',
              data: cleanBase64,
            },
          },
          { text: promptText },
        ],
      },
      config: {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            latex: {
              type: Type.STRING,
              description: 'The LaTeX representation of the handwritten math or symbols',
            },
            readableText: {
              type: Type.STRING,
              description: 'Plain text readable representation',
            },
            symbolsDetected: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'List of individual mathematical and alphanumeric symbols detected',
            },
            isSolvable: {
              type: Type.BOOLEAN,
              description: 'Whether this handwriting contains an equation or solvable arithmetic expression',
            },
            solution: {
              type: Type.STRING,
              description: 'The final evaluated answer or solved root(s)',
            },
            autoAnswer: {
              type: Type.STRING,
              description: 'The short answer to write directly next to the equal sign on the canvas',
            },
            steps: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: 'Step-by-step mathematical derivation or simplification',
            },
            isGraphable: {
              type: Type.BOOLEAN,
              description: 'Whether this can be plotted as a 2D function curve y = f(x)',
            },
            graphJsExpr: {
              type: Type.STRING,
              description: 'JavaScript expression of x (e.g., Math.pow(x, 2) - 4) for client-side plotting',
            },
            graphTitle: {
              type: Type.STRING,
              description: 'Title for the graph (e.g. y = x^2 - 4)',
            },
          },
          required: ['latex', 'readableText', 'symbolsDetected', 'isSolvable'],
        },
      },
    });

    const jsonText = response.text?.trim() || '{}';
    let parsedResult;
    try {
      parsedResult = JSON.parse(jsonText);
    } catch {
      parsedResult = {
        latex: '\\text{Unrecognized handwriting}',
        readableText: 'Unrecognized handwriting',
        symbolsDetected: [],
        isSolvable: false,
      };
    }

    return res.json(parsedResult);
  } catch (err: any) {
    console.error('Error in /api/math/recognize:', err);
    return res.status(500).json({
      error: 'Failed to recognize math handwriting',
      details: err?.message || String(err),
    });
  }
});

// Start Express server and mount Vite
async function startServer() {
  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(port), '0.0.0.0', () => {
    console.log(`Server listening on http://0.0.0.0:${port}`);
  });
}

startServer();
