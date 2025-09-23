type Props = {
  url: string | null
}

export default function Preview({ url }: Props) {
  if (!url) return null
  return (
    <div style={{ marginTop: 12 }}>
      <img src={url} alt="preview" style={{ maxWidth: 320, borderRadius: 8 }} />
    </div>
  )
}

