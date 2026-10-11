import { createRoot } from 'react-dom/client';
import 'katex/dist/katex.min.css';
import '../index.css';
import LabApp from './LabApp';

createRoot(document.getElementById('root')!).render(<LabApp />);
