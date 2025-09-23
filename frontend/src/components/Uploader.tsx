import { useRef, useState } from 'react'

type Props = {
  onFileSelected: (file: File | null) => void
  previewUrl?: string | null
}

export default function Uploader({ onFileSelected, previewUrl }: Props) {
  const inputRef = useRef<HTMLInputElement | null>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleFileSelect = (file: File | null) => {
    onFileSelected(file)
    if (!file && inputRef.current) inputRef.current.value = ''
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file)
    }
  }

  const handleClick = () => {
    inputRef.current?.click()
  }

  return (
    <div
      className={`upload-tile ${isDragOver ? 'drag-over' : ''} ${previewUrl ? 'has-image' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        onChange={(e) => {
          const f = e.target.files?.[0] ?? null
          handleFileSelect(f)
        }}
        style={{ display: 'none' }}
      />
      {previewUrl ? (
        <div className="image-preview">
          <img src={previewUrl} alt="Uploaded plant" />
          <div className="image-overlay">
            <div className="upload-icon">📁</div>
            <p>Click to change image</p>
          </div>
        </div>
      ) : (
        <div className="upload-content">
          <div className="upload-icon">📁</div>
          <h3>Upload Plant Image</h3>
          <p>Click to browse or drag and drop an image here</p>
          <p className="upload-formats">Supports: JPG, PNG, GIF, WebP</p>
        </div>
      )}
    </div>
  )
}

