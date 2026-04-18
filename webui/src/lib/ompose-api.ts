export type TargetKeypoint = {
  x: number
  y: number
  visibility?: number | null
  note?: string | null
}

export type PoseLandmark = {
  index: number
  name: string
  x: number
  y: number
  z?: number | null
  pixel_x: number
  pixel_y: number
  visibility?: number | null
  presence?: number | null
}

export type PoseRecommendation = {
  name: string
  category: string
  description: string
  reasoning: string
  keypoint_adjustments: Record<string, string>
  difficulty: string
  camera_angle_suggestion: string
  marker_template: string
  marker_side: string
  marker_intensity: number
  target_keypoints: Record<string, TargetKeypoint>
  target_pose_quality_notes?: string | null
  pose_template_id?: string | null
  guide_params?: Record<string, unknown>
  correction_focus?: string[]
}

export type PoseGuideRecommendation = {
  pose_template_id: string
  name: string
  category: string
  description: string
  reasoning: string
  difficulty: string
  camera_angle_suggestion: string
  guide_params: Record<string, unknown>
  correction_focus: string[]
}

export type PipelineResult = {
  input_image: string
  overlay_image: string
  contact_sheet_image: string
  result_json: string
  scene: {
    scene_type: string
    location_category: string
    lighting: string
    mood: string
    composition_notes: string
  }
  recommendations: PoseRecommendation[]
}

export type PoseGuideResult = {
  input_image: string
  scene: PipelineResult['scene']
  current_pose: PoseData
  selected_pose_guide: PoseGuideRecommendation
  target_pose: PoseRecommendation
  usage: Record<string, unknown>
}

export type PoseData = {
  image_width: number
  image_height: number
  keypoint_profile: string
  keypoints: Record<string, PoseLandmark>
  confidence: number
}

export type PoseResultMessage = {
  type: 'pose_result'
  frame_id: number
  latency_ms: number
  inference_ms: number | null
  dropped_frames: number
  current_pose: PoseData | null
  score: number | null
  corrections: string[]
  has_overlay: boolean
  pose_stale?: boolean
  warning?: string
  error?: string
}

export type RecommendationJob = {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  result: PipelineResult | null
  error: string | null
}

export type PoseGuideJob = {
  job_id: string
  status: 'queued' | 'running' | 'completed' | 'failed'
  result: PoseGuideResult | null
  error: string | null
}

export type ModelInfo = {
  backend: string
  model_path: string
  model_loaded: boolean
  device: string
  imgsz: number
  conf: number
  max_det: number
  keypoint_profile: string
}

const API_URL = import.meta.env.VITE_OMPOSE_API_URL ?? 'http://127.0.0.1:8080'
const WS_URL = import.meta.env.VITE_OMPOSE_WS_URL ?? 'ws://127.0.0.1:8080/ws/pose'

export function createPoseWebSocket() {
  return new WebSocket(WS_URL)
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await fetch(`${API_URL}/models/current`)
  if (!response.ok) throw new Error(`Model request failed: ${response.status}`)
  return response.json()
}

export async function requestPoseRecommendation(
  image: Blob,
  options: { recommendationCount: number; poseIndex: number },
): Promise<PipelineResult> {
  const body = new FormData()
  body.append('file', image, 'snapshot.jpg')
  body.append('recommendation_count', String(options.recommendationCount))
  body.append('pose_index', String(options.poseIndex))

  const response = await fetch(`${API_URL}/api/recommendations/upload`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Recommendation request failed: ${response.status}`)
  }
  return response.json()
}

export async function startPoseRecommendationJob(
  image: Blob,
  options: { recommendationCount: number; poseIndex: number },
): Promise<RecommendationJob> {
  const body = new FormData()
  body.append('file', image, 'snapshot.jpg')
  body.append('recommendation_count', String(options.recommendationCount))
  body.append('pose_index', String(options.poseIndex))

  const response = await fetch(`${API_URL}/api/recommendations/upload/async`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Recommendation job request failed: ${response.status}`)
  }
  return response.json()
}

export async function getPoseRecommendationJob(jobId: string): Promise<RecommendationJob> {
  const response = await fetch(`${API_URL}/api/recommendations/jobs/${jobId}`)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Recommendation job status failed: ${response.status}`)
  }
  return response.json()
}

export async function startPoseGuideJob(
  image: Blob,
  options: { poseIndex: number },
): Promise<PoseGuideJob> {
  const body = new FormData()
  body.append('file', image, 'snapshot.jpg')
  body.append('pose_index', String(options.poseIndex))

  const response = await fetch(`${API_URL}/api/pose-guide/upload/async`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Pose guide job request failed: ${response.status}`)
  }
  return response.json()
}

export async function getPoseGuideJob(jobId: string): Promise<PoseGuideJob> {
  const response = await fetch(`${API_URL}/api/pose-guide/jobs/${jobId}`)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Pose guide job status failed: ${response.status}`)
  }
  return response.json()
}

// ── Capture Photo ────────────────────────────────────────────────

export type CaptureRecord = {
  capture_id: string
  data_url: string
  mime_type: string
  size_bytes: number
  captured_at: string
}

export async function capturePhoto(image: Blob): Promise<CaptureRecord> {
  const body = new FormData()
  body.append('file', image, 'capture.jpg')

  const response = await fetch(`${API_URL}/api/capture`, {
    method: 'POST',
    body,
  })

  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Capture request failed: ${response.status}`)
  }
  return response.json()
}

export async function listCaptures(): Promise<{ captures: CaptureRecord[]; count: number }> {
  const response = await fetch(`${API_URL}/api/captures`)
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `List captures failed: ${response.status}`)
  }
  return response.json()
}

export async function deleteCapture(captureId: string): Promise<void> {
  const response = await fetch(`${API_URL}/api/captures/${captureId}`, { method: 'DELETE' })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Delete capture failed: ${response.status}`)
  }
}

export async function clearCaptures(): Promise<void> {
  const response = await fetch(`${API_URL}/api/captures`, { method: 'DELETE' })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(text || `Clear captures failed: ${response.status}`)
  }
}
