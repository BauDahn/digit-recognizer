/** What the panel says about the network's learning. */
export interface LessonSummary {
  learning: boolean;
  count: number;
  headline: string;
  detail?: string;
  onForget: () => void;
}
