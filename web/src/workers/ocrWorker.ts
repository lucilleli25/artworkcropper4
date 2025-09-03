/* eslint-disable no-restricted-globals */
import { createWorker } from 'tesseract.js'

let tessWorker: any | null = null
let initPromise: Promise<any> | null = null

async function getWorker() {
  if (tessWorker) return tessWorker
  if (!initPromise) {
    initPromise = (async () => {
      const w: any = await (createWorker as any)()
      if (w.loadLanguage) {
        await w.loadLanguage('eng')
      }
      if (w.initialize) {
        await w.initialize('eng')
      } else if (w.reinitialize) {
        await w.reinitialize('eng')
      }
      if (w.setParameters) {
        await w.setParameters({
          tessedit_char_whitelist: '0123456789-',
          tessedit_pageseg_mode: '7',
          classify_bln_numeric_mode: '1',
          user_defined_dpi: '300',
        } as any)
      }
      tessWorker = w
      return w
    })()
  }
  return initPromise
}

type OCRJob = { id: string; bitmap: ImageBitmap }

self as any

function otsuThreshold(gray: Uint8Array): number {
  const hist = new Array(256).fill(0)
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++
  const total = gray.length
  let sum = 0
  for (let t = 0; t < 256; t++) sum += t * hist[t]
  let sumB = 0
  let wB = 0
  let varMax = -1
  let threshold = 0
  for (let t = 0; t < 256; t++) {
    wB += hist[t]
    if (wB === 0) continue
    const wF = total - wB
    if (wF === 0) break
    sumB += t * hist[t]
    const mB = sumB / wB
    const mF = (sum - sumB) / wF
    const between = wB * wF * (mB - mF) * (mB - mF)
    if (between > varMax) { varMax = between; threshold = t }
  }
  return threshold
}

function preprocessBitmapToBW(bitmap: ImageBitmap, scale = 3): OffscreenCanvas {
  const w = Math.max(1, Math.floor(bitmap.width * scale))
  const h = Math.max(1, Math.floor(bitmap.height * scale))
  const off = new OffscreenCanvas(w, h)
  const ctx = off.getContext('2d')!
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)
  ;(ctx as any).imageSmoothingEnabled = false
  ctx.drawImage(bitmap, 0, 0, w, h)
  const img = ctx.getImageData(0, 0, w, h)
  const data = img.data
  const gray = new Uint8Array(w * h)
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    gray[j] = (0.299 * r + 0.587 * g + 0.114 * b) | 0
  }
  const otsu = otsuThreshold(gray)
  const T = Math.min(245, otsu + 12) // lean brighter to catch light-gray text
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const v = gray[j] < T ? 0 : 255
    data[i] = v; data[i + 1] = v; data[i + 2] = v; data[i + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
  return off
}

self.onmessage = (async (e: MessageEvent) => {
  const data = e.data as { jobs: OCRJob[] }
  const jobs = data?.jobs || []
  const w = await getWorker()
  for (const j of jobs) {
    try {
      // Pass 1: 3x scaled, binarized, psm 7
      const bw1 = preprocessBitmapToBW(j.bitmap, 3)
      const res = await w.recognize(bw1, 'eng', {
        tessedit_char_whitelist: '0123456789-',
        tessedit_pageseg_mode: '7',
        user_defined_dpi: '300',
        preserve_interword_spaces: '1',
      } as any)
      let text = (res?.data?.text || '').trim()
      let confidence = res?.data?.confidence || 0
      const ok = /^\d{3,4}-\d{2}$/.test(text) && confidence >= 70
      if (!ok) {
        // Pass 2: 4x scaled, slightly brighter threshold, psm 8
        const bw2 = preprocessBitmapToBW(j.bitmap, 4)
        const res2 = await w.recognize(bw2, 'eng', {
          tessedit_char_whitelist: '0123456789-',
          tessedit_pageseg_mode: '8',
          user_defined_dpi: '300',
          preserve_interword_spaces: '1',
        } as any)
        const t2 = (res2?.data?.text || '').trim()
        const c2 = res2?.data?.confidence || 0
        if (/^\d{3,4}-\d{2}$/.test(t2) && c2 >= Math.max(60, confidence)) {
          text = t2; confidence = c2
        }
      }
      ;(self as unknown as Worker).postMessage({ type: 'ocr-result', id: j.id, text, confidence })
    } catch (err) {
      ;(self as unknown as Worker).postMessage({ type: 'ocr-result', id: j.id, text: '', confidence: 0 })
    }
  }
}) as any


