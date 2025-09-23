export type Prediction = { label: string; confidence: number }
export type Remedy = { label: string; actions: string[] }
export type AnalyzeResponse = { filename: string; predictions: Prediction[]; remedies: Remedy[] }

