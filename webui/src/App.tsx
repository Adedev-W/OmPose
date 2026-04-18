import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Activity,
  Camera,
  CameraOff,
  CircleStop,
  Download,
  ImageIcon,
  Loader2,
  PlugZap,
  RefreshCcw,
  Sparkles,
  Trash2,
  Wifi,
  WifiOff,
  X,
} from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Panel } from '@/components/ui/panel'
import {
  capturePhoto,
  clearCaptures,
  createPoseWebSocket,
  deleteCapture,
  getPoseGuideJob,
  getModelInfo,
  startPoseGuideJob,
  type CaptureRecord,
  type ModelInfo,
  type PoseGuideResult,
  type PoseData,
  type PoseRecommendation,
  type PoseResultMessage,
  type TargetKeypoint,
} from '@/lib/ompose-api'

const STREAM_FPS = Number(import.meta.env.VITE_STREAM_FPS ?? 12)
const JPEG_QUALITY = Number(import.meta.env.VITE_JPEG_QUALITY ?? 0.6)
const SERVER_OVERLAY_DEFAULT = import.meta.env.VITE_SERVER_OVERLAY === 'true'
const CANVAS_WIDTH = 720
const CANVAS_HEIGHT = 960
const POSE_HOLD_MS = 1400
const SERVER_OVERLAY_HOLD_MS = 650
const TARGET_POSE_STORAGE_KEY = 'ompose.targetPose.v1'

const BODY_CONNECTIONS = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'],
  ['left_knee', 'left_ankle'],
  ['left_ankle', 'left_heel'],
  ['left_heel', 'left_foot_index'],
  ['right_hip', 'right_knee'],
  ['right_knee', 'right_ankle'],
  ['right_ankle', 'right_heel'],
  ['right_heel', 'right_foot_index'],
] as const
const BODY_JOINTS: ReadonlySet<string> = new Set(BODY_CONNECTIONS.flatMap(([start, end]) => [start, end]))

type CameraState = 'idle' | 'starting' | 'ready' | 'error'
type StreamState = 'disconnected' | 'connecting' | 'connected' | 'error'
type RecommendationStatus = 'idle' | 'queued' | 'running' | 'completed' | 'failed'

function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const socketRef = useRef<WebSocket | null>(null)
  const frameTimerRef = useRef<number | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const awaitingOverlayRef = useRef(false)
  const sendingFrameRef = useRef(false)
  const targetPoseRef = useRef<PoseRecommendation | null>(null)
  const latestPoseRef = useRef<PoseData | null>(null)
  const latestPoseUpdatedAtRef = useRef(0)
  const latestResultRef = useRef<PoseResultMessage | null>(null)
  const serverOverlayRef = useRef<ImageBitmap | null>(null)
  const serverOverlayUpdatedAtRef = useRef(0)
  const returnOverlayRef = useRef(SERVER_OVERLAY_DEFAULT)

  const [cameraState, setCameraState] = useState<CameraState>('idle')
  const [streamState, setStreamState] = useState<StreamState>('disconnected')
  const [error, setError] = useState<string | null>(null)
  const [latestResult, setLatestResult] = useState<PoseResultMessage | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [triggering, setTriggering] = useState(false)
  const [recommendationStatus, setRecommendationStatus] = useState<RecommendationStatus>('idle')
  const [recommendationJobId, setRecommendationJobId] = useState<string | null>(null)
  const [targetPose, setTargetPose] = useState<PoseRecommendation | null>(null)
  const [lastRecommendation, setLastRecommendation] = useState<PoseGuideResult | null>(null)
  const [returnOverlay, setReturnOverlay] = useState(SERVER_OVERLAY_DEFAULT)
  const [sentFrames, setSentFrames] = useState(0)
  const [captures, setCaptures] = useState<CaptureRecord[]>([])
  const [capturing, setCapturing] = useState(false)
  const [lightboxSrc, setLightboxSrc] = useState<string | null>(null)

  const isCameraReady = cameraState === 'ready'
  const isConnected = streamState === 'connected'
  const scorePercent = latestResult?.score == null ? null : Math.round(latestResult.score * 100)

  const drawLiveCanvas = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!canvas) return

    const context = canvas.getContext('2d')
    if (!context) return

    const now = performance.now()
    const hasVideo = video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
    const hasFreshServerOverlay =
      returnOverlayRef.current &&
      serverOverlayRef.current &&
      now - serverOverlayUpdatedAtRef.current < SERVER_OVERLAY_HOLD_MS

    if (hasFreshServerOverlay && serverOverlayRef.current) {
      context.drawImage(serverOverlayRef.current, 0, 0, canvas.width, canvas.height)
      return
    }

    if (hasVideo && video) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
    } else {
      fillCanvas(context, canvas.width, canvas.height, '#111')
    }

    const latestPose = latestPoseRef.current
    const poseAge = now - latestPoseUpdatedAtRef.current
    if (latestPose && poseAge < POSE_HOLD_MS) {
      drawPoseSkeleton(context, latestPose, {
        color: latestResultRef.current?.pose_stale ? '#f4c95d' : '#86a6ff',
        alpha: latestResultRef.current?.pose_stale ? 0.62 : 0.78,
        lineWidth: 3,
        radius: 5,
      })
    }

    if (targetPoseRef.current) {
      drawTargetSkeleton(context, targetPoseRef.current.target_keypoints, {
        color: '#31f795',
        alpha: 0.9,
        lineWidth: 4,
        radius: 6,
      })
    }
  }, [])

  const startDrawLoop = useCallback(() => {
    if (animationFrameRef.current != null) return
    const tick = () => {
      drawLiveCanvas()
      animationFrameRef.current = window.requestAnimationFrame(tick)
    }
    tick()
  }, [drawLiveCanvas])

  const stopDrawLoop = useCallback(() => {
    if (animationFrameRef.current != null) {
      window.cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }, [])

  const captureFrameBlob = useCallback(async (quality = JPEG_QUALITY) => {
    const video = videoRef.current
    const canvas = captureCanvasRef.current
    if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return null

    const context = canvas.getContext('2d')
    if (!context) return null
    context.drawImage(video, 0, 0, canvas.width, canvas.height)

    return new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', quality)
    })
  }, [])

  const startCamera = useCallback(async () => {
    setError(null)
    setCameraState('starting')
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 720 },
          height: { ideal: 960 },
        },
      })
      streamRef.current = mediaStream
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        await videoRef.current.play()
      }
      setCameraState('ready')
      startDrawLoop()
    } catch (cause) {
      setCameraState('error')
      setError(cause instanceof Error ? cause.message : 'Unable to open camera.')
    }
  }, [startDrawLoop])

  const stopFrameLoop = useCallback(() => {
    if (frameTimerRef.current != null) {
      window.clearInterval(frameTimerRef.current)
      frameTimerRef.current = null
    }
  }, [])

  const stopStream = useCallback(() => {
    stopFrameLoop()
    awaitingOverlayRef.current = false
    socketRef.current?.close()
    socketRef.current = null
    setStreamState('disconnected')
  }, [stopFrameLoop])

  const stopCamera = useCallback(() => {
    stopStream()
    stopDrawLoop()
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    latestPoseRef.current = null
    latestResultRef.current = null
    setCameraState('idle')
    setLatestResult(null)
    drawBlankCanvas(canvasRef.current, 'Camera off')
  }, [stopDrawLoop, stopStream])

  const sendFrame = useCallback(async () => {
    const socket = socketRef.current
    if (!socket || socket.readyState !== WebSocket.OPEN || sendingFrameRef.current) return
    if (returnOverlayRef.current && awaitingOverlayRef.current) return

    sendingFrameRef.current = true
    try {
      const blob = await captureFrameBlob(JPEG_QUALITY)
      if (!blob || socket.readyState !== WebSocket.OPEN) return
      socket.send(blob)
      setSentFrames((value) => value + 1)
    } finally {
      sendingFrameRef.current = false
    }
  }, [captureFrameBlob])

  const connectStream = useCallback(() => {
    if (!isCameraReady || socketRef.current) return
    setError(null)
    setStreamState('connecting')

    const socket = createPoseWebSocket()
    socket.binaryType = 'blob'
    socketRef.current = socket

    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: 'config',
        return_overlay: returnOverlayRef.current,
        jpeg_quality: Math.round(JPEG_QUALITY * 100),
        max_fps: STREAM_FPS,
        target_pose: targetPoseRef.current,
      }))
    }

    socket.onmessage = async (event) => {
      if (typeof event.data === 'string') {
        const message = JSON.parse(event.data)
        if (message.type === 'ready' || message.type === 'config_updated') {
          setStreamState('connected')
          stopFrameLoop()
          frameTimerRef.current = window.setInterval(sendFrame, 1000 / STREAM_FPS)
          return
        }
        if (message.type === 'target_pose_updated') return
        if (message.type === 'error') {
          setError(message.message ?? 'Streaming error.')
          setStreamState('error')
          return
        }
        if (message.type === 'pose_result') {
          const result = message as PoseResultMessage
          latestResultRef.current = result
          setLatestResult(result)
          if (result.current_pose) {
            latestPoseRef.current = result.current_pose
            latestPoseUpdatedAtRef.current = performance.now()
          }
          awaitingOverlayRef.current = returnOverlayRef.current && Boolean(result.has_overlay)
        }
        return
      }

      if (event.data instanceof Blob) {
        const nextBitmap = await createImageBitmap(event.data)
        serverOverlayRef.current?.close()
        serverOverlayRef.current = nextBitmap
        serverOverlayUpdatedAtRef.current = performance.now()
        awaitingOverlayRef.current = false
      }
    }

    socket.onerror = () => {
      setError('WebSocket connection failed.')
      setStreamState('error')
      stopFrameLoop()
    }

    socket.onclose = () => {
      socketRef.current = null
      stopFrameLoop()
      awaitingOverlayRef.current = false
      setStreamState((state) => state === 'error' ? state : 'disconnected')
    }
  }, [isCameraReady, sendFrame, stopFrameLoop])

  const updateOverlayMode = useCallback((enabled: boolean) => {
    setReturnOverlay(enabled)
    returnOverlayRef.current = enabled
    awaitingOverlayRef.current = false
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({
        type: 'config',
        return_overlay: enabled,
        jpeg_quality: Math.round(JPEG_QUALITY * 100),
        max_fps: STREAM_FPS,
      }))
    }
  }, [])

  const applyTargetPose = useCallback((nextPose: PoseRecommendation | null, persist = true) => {
    setTargetPose(nextPose)
    targetPoseRef.current = nextPose
    if (persist) {
      if (nextPose) {
        window.sessionStorage.setItem(TARGET_POSE_STORAGE_KEY, JSON.stringify(nextPose))
      } else {
        window.sessionStorage.removeItem(TARGET_POSE_STORAGE_KEY)
      }
    }
    const socket = socketRef.current
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: 'target_pose', target_pose: nextPose }))
    }
  }, [])

  const triggerPose = useCallback(async () => {
    setError(null)
    setTriggering(true)
    setRecommendationStatus('queued')
    setRecommendationJobId(null)
    try {
      const blob = await captureFrameBlob(0.82)
      if (!blob) throw new Error('No camera frame is available.')

      const queued = await startPoseGuideJob(blob, {
        poseIndex: 0,
      })
      setRecommendationJobId(queued.job_id)

      const result = await waitForPoseGuide(queued.job_id, setRecommendationStatus)
      const nextPose = result.target_pose
      if (!nextPose) throw new Error('Backend returned no target pose.')

      setLastRecommendation(result)
      applyTargetPose(nextPose)
      setRecommendationStatus('completed')
    } catch (cause) {
      setRecommendationStatus('failed')
      setError(cause instanceof Error ? cause.message : 'Trigger Pose failed.')
    } finally {
      setTriggering(false)
    }
  }, [applyTargetPose, captureFrameBlob])

  const handleCapture = useCallback(async () => {
    if (capturing) return
    setCapturing(true)
    setError(null)
    try {
      const blob = await captureFrameBlob(0.92)
      if (!blob) throw new Error('No camera frame available.')
      const record = await capturePhoto(blob)
      setCaptures((prev) => [record, ...prev])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Capture failed.')
    } finally {
      setCapturing(false)
    }
  }, [captureFrameBlob, capturing])

  const handleDeleteCapture = useCallback(async (captureId: string) => {
    try {
      await deleteCapture(captureId)
      setCaptures((prev) => prev.filter((c) => c.capture_id !== captureId))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Delete failed.')
    }
  }, [])

  const handleClearCaptures = useCallback(async () => {
    try {
      await clearCaptures()
      setCaptures([])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Clear failed.')
    }
  }, [])

  const downloadCapture = useCallback((record: CaptureRecord) => {
    const link = document.createElement('a')
    link.href = record.data_url
    link.download = `ompose-${record.capture_id}.jpg`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }, [])

  useEffect(() => {
    drawBlankCanvas(canvasRef.current, 'Start camera')
    const storedTarget = loadStoredTargetPose()
    if (storedTarget) {
      setTargetPose(storedTarget)
      targetPoseRef.current = storedTarget
    }
    getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
    return () => {
      stopFrameLoop()
      stopDrawLoop()
      streamRef.current?.getTracks().forEach((track) => track.stop())
      socketRef.current?.close()
      serverOverlayRef.current?.close()
    }
  }, [stopDrawLoop, stopFrameLoop])

  return (
    <main className="min-h-svh bg-[#f6f5ef] text-[#151515]">
      <div className="mx-auto flex min-h-svh w-full max-w-[1440px] flex-col gap-4 px-4 py-4 lg:px-6">
        <header className="flex flex-col gap-3 border-b border-[#d7d5ca] pb-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-[#6d6a5f]">OmPose Live</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-normal text-[#111] md:text-3xl">
              Pose director camera
            </h1>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge label="Camera" state={cameraState} />
            <StatusBadge label="Stream" state={streamState} />
            <Badge variant="outline">{modelInfo?.model_path?.split('/').at(-1) ?? 'model unknown'}</Badge>
          </div>
        </header>

        <section className="grid flex-1 gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="relative min-h-[520px] overflow-hidden border border-[#d7d5ca] bg-black">
            <video ref={videoRef} className="hidden" muted playsInline />
            <canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              className="h-full min-h-[520px] w-full object-contain"
            />
            <canvas ref={captureCanvasRef} width={CANVAS_WIDTH} height={CANVAS_HEIGHT} className="hidden" />
            <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
              <Metric label="Latency" value={formatMs(latestResult?.latency_ms)} />
              <Metric label="Infer" value={formatMs(latestResult?.inference_ms)} />
              <Metric label="Dropped" value={`${latestResult?.dropped_frames ?? 0}`} />
              <Metric label="Sent" value={`${sentFrames}`} />
              {latestResult?.pose_stale && <Metric label="Hold" value="pose" />}
            </div>
            {scorePercent != null && (
              <div className="pointer-events-none absolute bottom-3 left-3 rounded-md border border-white/20 bg-black/70 px-3 py-2 text-white">
                <div className="text-xs uppercase tracking-[0.18em] text-white/60">Pose match</div>
                <div className="text-2xl font-semibold">{scorePercent}%</div>
              </div>
            )}
            {/* Capture shutter button */}
            {isCameraReady && (
              <button
                id="capture-shutter-btn"
                onClick={handleCapture}
                disabled={capturing}
                title="Capture photo"
                className="absolute bottom-4 right-4 flex size-14 items-center justify-center rounded-full border-[3px] border-white/80 bg-white/20 text-white shadow-lg backdrop-blur-md transition-all duration-200 hover:scale-110 hover:bg-white/35 active:scale-95 disabled:opacity-50"
                style={{ cursor: capturing ? 'wait' : 'pointer' }}
              >
                {capturing ? (
                  <Loader2 className="size-6 animate-spin" />
                ) : (
                  <div className="size-10 rounded-full bg-white shadow-inner" />
                )}
              </button>
            )}
          </div>

          <aside className="flex flex-col gap-4">
            <Panel>
              <div className="flex items-center justify-between gap-3">
                <h2 className="text-lg font-semibold">Controls</h2>
                <Activity className="size-5 text-[#2f7c62]" />
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2">
                <Button onClick={startCamera} disabled={cameraState === 'starting' || isCameraReady}>
                  {cameraState === 'starting' ? <Loader2 className="size-4 animate-spin" /> : <Camera className="size-4" />}
                  Start
                </Button>
                <Button variant="secondary" onClick={stopCamera} disabled={cameraState === 'idle'}>
                  <CameraOff className="size-4" />
                  Stop
                </Button>
                <Button onClick={connectStream} disabled={!isCameraReady || streamState === 'connecting' || isConnected}>
                  {streamState === 'connecting' ? <Loader2 className="size-4 animate-spin" /> : <PlugZap className="size-4" />}
                  Connect
                </Button>
                <Button variant="secondary" onClick={stopStream} disabled={streamState === 'disconnected'}>
                  <CircleStop className="size-4" />
                  Disconnect
                </Button>
              </div>
              <Button className="mt-3 w-full" onClick={triggerPose} disabled={!isCameraReady || triggering}>
                {triggering ? <Loader2 className="size-4 animate-spin" /> : <Sparkles className="size-4" />}
                {triggering ? 'Generating target' : 'Trigger Pose'}
              </Button>
              <label className="mt-4 flex items-center justify-between rounded-md border border-[#d7d5ca] bg-white px-3 py-2 text-sm">
                <span>Server overlay</span>
                <input
                  type="checkbox"
                  checked={returnOverlay}
                  onChange={(event) => updateOverlayMode(event.target.checked)}
                  className="size-4 accent-[#2f7c62]"
                />
              </label>
            </Panel>

            <Panel>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Live result</h2>
                {isConnected ? <Wifi className="size-5 text-[#2f7c62]" /> : <WifiOff className="size-5 text-[#7b2d26]" />}
              </div>
              <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <Info label="Frame" value={latestResult?.frame_id ?? '-'} />
                <Info label="Score" value={scorePercent != null ? `${scorePercent}%` : '-'} />
                <Info label="Keypoints" value={latestResult?.current_pose ? Object.keys(latestResult.current_pose.keypoints).length : '-'} />
                <Info label="AI job" value={recommendationJobId ? recommendationStatus : '-'} />
              </dl>
              <div className="mt-4">
                <div className="text-xs font-semibold uppercase tracking-[0.18em] text-[#6d6a5f]">Corrections</div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {latestResult?.corrections?.length ? latestResult.corrections.map((item) => (
                    <Badge key={item} variant="secondary">{humanize(item)}</Badge>
                  )) : <span className="text-sm text-[#6d6a5f]">No target pose yet.</span>}
                </div>
              </div>
            </Panel>

            <Panel>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold">Target pose</h2>
                <Button variant="ghost" size="icon" onClick={() => applyTargetPose(null)} title="Clear target pose">
                  <RefreshCcw className="size-4" />
                </Button>
              </div>
              {targetPose ? (
                <div className="mt-3 space-y-2">
                  <div className="font-medium">{targetPose.name}</div>
                  <p className="text-sm text-[#6d6a5f]">{targetPose.description}</p>
                  <div className="flex gap-2">
                    <Badge>{targetPose.difficulty}</Badge>
                    <Badge variant="outline">{targetPose.pose_template_id ?? targetPose.marker_template}</Badge>
                  </div>
                </div>
              ) : (
                <p className="mt-3 text-sm text-[#6d6a5f]">Target skeleton will appear after Trigger Pose.</p>
              )}
              {lastRecommendation?.scene && (
                <div className="mt-4 border-t border-[#d7d5ca] pt-3 text-sm text-[#6d6a5f]">
                  Scene: {lastRecommendation.scene.location_category} / {lastRecommendation.scene.lighting}
                </div>
              )}
            </Panel>

            <Panel>
              <div className="flex items-center justify-between">
                <h2 className="text-lg font-semibold flex items-center gap-2">
                  <ImageIcon className="size-5 text-[#2f7c62]" />
                  Captures
                  {captures.length > 0 && (
                    <Badge variant="secondary">{captures.length}</Badge>
                  )}
                </h2>
                {captures.length > 0 && (
                  <Button variant="ghost" size="icon" onClick={handleClearCaptures} title="Clear all captures">
                    <Trash2 className="size-4 text-[#7b2d26]" />
                  </Button>
                )}
              </div>
              {captures.length === 0 ? (
                <p className="mt-3 text-sm text-[#6d6a5f]">
                  No captures yet. Use the shutter button on the camera to take photos.
                </p>
              ) : (
                <div className="mt-3 grid grid-cols-3 gap-2">
                  {captures.map((record) => (
                    <div
                      key={record.capture_id}
                      className="group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-md border border-[#d7d5ca] bg-black transition-all duration-200 hover:border-[#2f7c62] hover:shadow-md"
                    >
                      <img
                        src={record.data_url}
                        alt={`Capture ${record.capture_id}`}
                        className="size-full object-cover transition-transform duration-300 group-hover:scale-105"
                        onClick={() => setLightboxSrc(record.data_url)}
                      />
                      <div className="absolute inset-x-0 bottom-0 flex items-center justify-between bg-gradient-to-t from-black/80 to-transparent px-1.5 py-1.5 opacity-0 transition-opacity duration-200 group-hover:opacity-100">
                        <button
                          onClick={(e) => { e.stopPropagation(); downloadCapture(record) }}
                          className="rounded p-1 text-white/80 hover:bg-white/20 hover:text-white"
                          title="Download"
                        >
                          <Download className="size-3.5" />
                        </button>
                        <span className="text-[10px] text-white/50">{formatBytes(record.size_bytes)}</span>
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeleteCapture(record.capture_id) }}
                          className="rounded p-1 text-white/80 hover:bg-red-500/30 hover:text-red-300"
                          title="Delete"
                        >
                          <Trash2 className="size-3.5" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Panel>

            {error && (
              <div className="rounded-md border border-[#c66b5d] bg-[#fff2ef] px-3 py-2 text-sm text-[#7b2d26]">
                {error}
              </div>
            )}
          </aside>
        </section>
      </div>

      {/* Lightbox modal */}
      {lightboxSrc && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm"
          onClick={() => setLightboxSrc(null)}
        >
          <button
            className="absolute right-4 top-4 rounded-full bg-white/15 p-2 text-white transition-colors hover:bg-white/30"
            onClick={() => setLightboxSrc(null)}
            title="Close"
          >
            <X className="size-6" />
          </button>
          <img
            src={lightboxSrc}
            alt="Captured photo"
            className="max-h-[90vh] max-w-[90vw] rounded-lg object-contain shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </main>
  )
}

async function waitForPoseGuide(
  jobId: string,
  setStatus: (status: RecommendationStatus) => void,
): Promise<PoseGuideResult> {
  const startedAt = Date.now()
  while (Date.now() - startedAt < 90_000) {
    await sleep(900)
    const job = await getPoseGuideJob(jobId)
    setStatus(job.status)
    if (job.status === 'completed' && job.result) return job.result
    if (job.status === 'failed') throw new Error(job.error ?? 'Pose guide job failed.')
  }
  throw new Error('Pose guide job timed out.')
}

function loadStoredTargetPose(): PoseRecommendation | null {
  try {
    const value = window.sessionStorage.getItem(TARGET_POSE_STORAGE_KEY)
    return value ? JSON.parse(value) as PoseRecommendation : null
  } catch {
    window.sessionStorage.removeItem(TARGET_POSE_STORAGE_KEY)
    return null
  }
}

function StatusBadge({ label, state }: { label: string; state: string }) {
  const good = state === 'ready' || state === 'connected'
  const pending = state === 'starting' || state === 'connecting'
  return (
    <Badge variant={good ? 'default' : pending ? 'secondary' : 'outline'}>
      {label}: {state}
    </Badge>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-white/15 bg-black/65 px-2 py-1 text-white">
      <div className="text-[10px] uppercase tracking-[0.18em] text-white/55">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  )
}

function Info({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md bg-[#f0efe7] px-3 py-2">
      <dt className="text-xs uppercase tracking-[0.18em] text-[#6d6a5f]">{label}</dt>
      <dd className="mt-1 font-mono text-sm text-[#151515]">{value}</dd>
    </div>
  )
}

function formatMs(value?: number | null) {
  if (value == null) return '-'
  return `${Math.round(value)}ms`
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes}B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)}KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)}MB`
}

function humanize(value: string) {
  return value.replaceAll('_', ' ')
}

function sleep(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function fillCanvas(context: CanvasRenderingContext2D, width: number, height: number, color: string) {
  context.fillStyle = color
  context.fillRect(0, 0, width, height)
}

function drawBlankCanvas(canvas: HTMLCanvasElement | null, label: string) {
  if (!canvas) return
  const context = canvas.getContext('2d')
  if (!context) return
  fillCanvas(context, canvas.width, canvas.height, '#111')
  context.fillStyle = '#f5f5f0'
  context.font = '24px ui-monospace, monospace'
  context.textAlign = 'center'
  context.fillText(label, canvas.width / 2, canvas.height / 2)
}

function drawPoseSkeleton(
  context: CanvasRenderingContext2D,
  pose: PoseData,
  style: { color: string; alpha: number; lineWidth: number; radius: number },
) {
  const points: Record<string, { x: number; y: number }> = {}
  for (const [name, landmark] of Object.entries(pose.keypoints)) {
    if (!BODY_JOINTS.has(name)) continue
    if (landmark.visibility != null && landmark.visibility < 0.08) continue
    points[name] = {
      x: landmark.x * context.canvas.width,
      y: landmark.y * context.canvas.height,
    }
  }
  drawSkeleton(context, points, style)
}

function drawTargetSkeleton(
  context: CanvasRenderingContext2D,
  targetKeypoints: Record<string, TargetKeypoint>,
  style: { color: string; alpha: number; lineWidth: number; radius: number },
) {
  const points: Record<string, { x: number; y: number }> = {}
  for (const [name, keypoint] of Object.entries(targetKeypoints)) {
    if (!BODY_JOINTS.has(name)) continue
    points[name] = {
      x: keypoint.x * context.canvas.width,
      y: keypoint.y * context.canvas.height,
    }
  }
  drawSkeleton(context, points, style)
}

function drawSkeleton(
  context: CanvasRenderingContext2D,
  points: Record<string, { x: number; y: number }>,
  style: { color: string; alpha: number; lineWidth: number; radius: number },
) {
  context.save()
  context.globalAlpha = style.alpha
  context.lineCap = 'round'
  context.lineJoin = 'round'
  context.strokeStyle = style.color
  context.lineWidth = style.lineWidth
  context.shadowColor = 'rgba(0, 0, 0, 0.45)'
  context.shadowBlur = 6

  for (const [start, end] of BODY_CONNECTIONS) {
    const first = points[start]
    const second = points[end]
    if (!first || !second) continue
    context.beginPath()
    context.moveTo(first.x, first.y)
    context.lineTo(second.x, second.y)
    context.stroke()
  }

  for (const point of Object.values(points)) {
    context.beginPath()
    context.fillStyle = '#ffffff'
    context.arc(point.x, point.y, style.radius, 0, Math.PI * 2)
    context.fill()
    context.beginPath()
    context.strokeStyle = style.color
    context.lineWidth = 2
    context.arc(point.x, point.y, style.radius + 1.5, 0, Math.PI * 2)
    context.stroke()
  }
  context.restore()
}

export default App
