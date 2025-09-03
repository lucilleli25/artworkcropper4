export async function fileToImageBitmap(file: File): Promise<ImageBitmap> {
  const arrayBuffer = await file.arrayBuffer()
  const blob = new Blob([arrayBuffer], { type: file.type })
  return await createImageBitmap(blob)
}

export function mapToOriginal(
  point: [number, number],
  scale: number
): [number, number] {
  return [point[0] / scale, point[1] / scale]
}

export function mapBboxToOriginal(
  bbox: { x: number; y: number; w: number; h: number },
  scale: number
): { x: number; y: number; w: number; h: number } {
  return {
    x: bbox.x / scale,
    y: bbox.y / scale,
    w: bbox.w / scale,
    h: bbox.h / scale,
  }
}

export function zeroPad(num: number, length: number): string {
  return String(num).padStart(length, '0')
}
