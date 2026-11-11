import React, { useEffect, useRef } from 'react';
import {
  Upload,
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Film,
  Trash2,
} from 'lucide-react';

interface ForensicVideoMonitorProps {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  onTogglePlay: () => void;
  onSeek: (timeS: number) => void;
  onFileUpload: (file: File) => void;
  onClearMedia: () => void;
  uploadedVideoUrl: string | null;
  uploadedFileName: string | null;
  rawPcmBuffer: Float32Array | null;
  sampleRate: number;
  isMuted: boolean;
  onToggleMute: () => void;
}

export const ForensicVideoMonitor: React.FC<ForensicVideoMonitorProps> = ({
  isPlaying,
  currentTime,
  duration,
  onTogglePlay,
  onSeek,
  onFileUpload,
  onClearMedia,
  uploadedVideoUrl,
  uploadedFileName,
  rawPcmBuffer,
  sampleRate,
  isMuted,
  onToggleMute,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Sync uploaded <video> element with audio playback state
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || !uploadedVideoUrl) return;
    if (Math.abs(vid.currentTime - currentTime) > 0.35) {
      vid.currentTime = currentTime;
    }
    if (isPlaying && vid.paused) {
      vid.play().catch(() => {});
    } else if (!isPlaying && !vid.paused) {
      vid.pause();
    }
  }, [isPlaying, currentTime, uploadedVideoUrl]);

  // Draw waveform preview when audio-only file is loaded
  useEffect(() => {
    if (uploadedVideoUrl || !rawPcmBuffer) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(240, rect.width);
    const h = Math.max(130, rect.height);
    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }

    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, w, h);

    // Center line
    const midY = h * 0.5;
    ctx.strokeStyle = '#1f1f22';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    // Waveform envelope
    const step = Math.max(1, Math.floor(rawPcmBuffer.length / w));
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const start = x * step;
      let maxAmp = 0;
      for (let j = 0; j < step && start + j < rawPcmBuffer.length; j += 4) {
        const a = Math.abs(rawPcmBuffer[start + j]);
        if (a > maxAmp) maxAmp = a;
      }
      const barH = Math.min(h * 0.44, maxAmp * h * 0.9);
      ctx.moveTo(x, midY - barH);
      ctx.lineTo(x, midY + barH);
    }
    ctx.stroke();

    // Playhead line
    if (duration > 0) {
      const px = (currentTime / duration) * w;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
      ctx.stroke();
    }

    ctx.restore();
  }, [rawPcmBuffer, uploadedVideoUrl, currentTime, duration]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) {
      onFileUpload(file);
    }
  };

  const formatTime = (sec: number) => {
    if (!Number.isFinite(sec) || sec < 0) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  };

  const hasMedia = Boolean(uploadedFileName || rawPcmBuffer);

  return (
    <div className="flex flex-col border-b border-[#1F1F1F] px-4 py-3.5">
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-[10px] uppercase tracking-[0.1em] text-[#737373] font-semibold">
          MEDIA
        </span>
        <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626]">
          {hasMedia && (
            <button
              type="button"
              onClick={onClearMedia}
              className="h-6 px-2 text-[11px] font-mono text-[#A1A1AA] hover:text-[#F43F5E] bg-[#111111] hover:bg-[#161616] rounded-none flex items-center gap-1 cursor-pointer transition-colors"
              title="Clear loaded file"
            >
              <Trash2 className="w-3 h-3" />
              Clear
            </button>
          )}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="h-6 px-2.5 text-[11px] font-medium bg-[#111111] hover:bg-[#161616] text-[#FAFAFA] rounded-none flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Upload className="w-3 h-3 text-[#FAFAFA]" />
            Upload File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*,audio/*,.wav,.mp3,.mp4,.webm,.mov"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload video or audio file"
          />
        </div>
      </div>

      {/* Video / Waveform / Dropzone Stage */}
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="relative w-full h-36 bg-[#000000] border border-[#262626] rounded-none overflow-hidden flex items-center justify-center"
      >
        {uploadedVideoUrl ? (
          <video
            ref={videoRef}
            src={uploadedVideoUrl}
            className="w-full h-full object-contain bg-black"
            muted
            playsInline
          />
        ) : rawPcmBuffer ? (
          <canvas
            ref={canvasRef}
            data-testid="forensic-video-canvas"
            className="w-full h-full block"
          />
        ) : (
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="w-full h-full flex flex-col items-center justify-center gap-1.5 text-[#A1A1AA] hover:text-[#FAFAFA] hover:bg-[#0A0A0A] transition-colors cursor-pointer rounded-none"
          >
            <Upload className="w-4 h-4 text-[#A1A1AA]" />
            <span className="text-[11px] font-medium text-[#FAFAFA]">
              Drop video or audio file here
            </span>
            <span className="text-[10px] font-mono text-[#737373] tracking-wider">
              MP4 · MOV · WEBM · WAV · MP3
            </span>
          </button>
        )}
      </div>

      {/* Transport & Timeline Scrubber Bar */}
      <div className="mt-2.5 flex items-center gap-1.5">
        <div className="inline-flex items-center border border-[#262626] divide-x divide-[#262626] shrink-0">
          <button
            type="button"
            onClick={onTogglePlay}
            disabled={!hasMedia}
            aria-label={isPlaying ? 'Pause playback' : 'Start playback'}
            className="w-6 h-6 rounded-none bg-[#FFFFFF] hover:bg-[#E5E5E5] disabled:bg-[#111111] disabled:text-[#52525B] text-[#000000] flex items-center justify-center transition-colors shrink-0 cursor-pointer"
          >
            {isPlaying ? (
              <Pause className="w-3 h-3 fill-current" />
            ) : (
              <Play className="w-3 h-3 fill-current ml-0.5" />
            )}
          </button>

          <button
            type="button"
            onClick={() => onSeek(0)}
            disabled={!hasMedia}
            aria-label="Reset playback to start"
            className="w-6 h-6 rounded-none bg-[#111111] hover:bg-[#161616] disabled:opacity-40 text-[#A1A1AA] hover:text-[#FAFAFA] flex items-center justify-center transition-colors shrink-0 cursor-pointer"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
        </div>

        <input
          type="range"
          min={0}
          max={Math.max(1, duration)}
          step={0.1}
          disabled={!hasMedia}
          value={Math.min(currentTime, Math.max(1, duration))}
          onChange={(e) => onSeek(parseFloat(e.target.value))}
          aria-label="Media playback timeline scrubber"
          className="flex-1 min-w-0 cursor-ew-resize disabled:opacity-40"
        />

        <button
          type="button"
          onClick={onToggleMute}
          disabled={!hasMedia}
          aria-label={isMuted ? 'Unmute audio' : 'Mute audio'}
          className="w-6 h-6 rounded-none bg-[#111111] hover:bg-[#161616] disabled:opacity-40 text-[#A1A1AA] hover:text-[#FAFAFA] border border-[#262626] flex items-center justify-center transition-colors shrink-0 cursor-pointer"
        >
          {isMuted ? (
            <VolumeX className="w-3 h-3 text-[#F43F5E]" />
          ) : (
            <Volume2 className="w-3 h-3 text-[#FAFAFA]" />
          )}
        </button>

        <span className="font-mono tabular-nums text-[10.5px] text-[#A1A1AA] whitespace-nowrap shrink-0">
          {formatTime(currentTime)} / {formatTime(hasMedia ? duration : 0)}
        </span>
      </div>

      {uploadedFileName && (
        <div className="mt-2 pt-1.5 border-t border-[#1F1F1F] flex items-center justify-between text-[11px] text-[#A1A1AA] font-mono tabular-nums">
          <span className="truncate max-w-[180px] text-[#FAFAFA]">
            <Film className="w-3 h-3 inline mr-1.5 text-[#737373]" />
            {uploadedFileName}
          </span>
          <span className="text-[#737373]">{sampleRate} Hz</span>
        </div>
      )}
    </div>
  );
};
