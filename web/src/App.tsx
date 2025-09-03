import { useState, useRef, useEffect } from 'react'
import './index.css'
import { LabelWithHelp } from './components/ui/LabelWithHelp'
import Tesseract from 'tesseract.js'

// Image preprocessing function
function preprocessImage(ctx: CanvasRenderingContext2D, width: number, height: number, params: EdgeParams): Uint8ClampedArray {
  const imageData = ctx.getImageData(0, 0, width, height)
  const data = imageData.data
  const processed = new Uint8ClampedArray(data.length)
  
  for (let i = 0; i < data.length; i += 4) {
    let r = data[i]
    let g = data[i + 1]
    let b = data[i + 2]
    let a = data[i + 3]
    
    // Apply brightness
    if (params.brightness !== 0) {
      r = Math.max(0, Math.min(255, r + params.brightness))
      g = Math.max(0, Math.min(255, g + params.brightness))
      b = Math.max(0, Math.min(255, b + params.brightness))
    }
    
    // Apply contrast
    if (params.contrastBoost !== 0) {
      const factor = (259 * (params.contrastBoost + 255)) / (255 * (259 - params.contrastBoost))
      r = Math.max(0, Math.min(255, factor * (r - 128) + 128))
      g = Math.max(0, Math.min(255, factor * (g - 128) + 128))
      b = Math.max(0, Math.min(255, factor * (b - 128) + 128))
    }
    
    // Apply gamma correction
    if (params.gamma !== 1.0) {
      r = Math.pow(r / 255, params.gamma) * 255
      g = Math.pow(g / 255, params.gamma) * 255
      b = Math.pow(b / 255, params.gamma) * 255
    }
    
    processed[i] = r
    processed[i + 1] = g
    processed[i + 2] = b
    processed[i + 3] = a
  }

  return processed
}

// Enhanced content detection function for whole tattoo detection
function detectContent(imageData: Uint8ClampedArray, x: number, y: number, w: number, h: number, imageWidth: number, params: EdgeParams): boolean {
  let darkPixels = 0
  let totalPixels = 0
  let edgePixels = 0
  let connectedComponents = 0
  let maxComponentSize = 0
  let currentComponentSize = 0
  
  // Create a visited array for connected component analysis
  const visited = new Array(w * h).fill(false)
  
  for (let py = y; py < y + h; py++) {
    for (let px = x; px < x + w; px++) {
      const i = (py * imageWidth + px) * 4
      const r = imageData[i]
      const g = imageData[i + 1]
      const b = imageData[i + 2]
      const gray = (r + g + b) / 3
      
      totalPixels++
      
      // Check for dark pixels (content)
      if (gray < params.threshold) {
        darkPixels++
        
        // Simple connected component analysis
        const localX = px - x
        const localY = py - y
        const localIndex = localY * w + localX
        
        if (!visited[localIndex]) {
          // Flood fill to find connected component size
          currentComponentSize = floodFill(imageData, visited, px, py, x, y, w, h, imageWidth, params.threshold)
          if (currentComponentSize > 0) {
            connectedComponents++
            maxComponentSize = Math.max(maxComponentSize, currentComponentSize)
          }
        }
      }
      
      // Check for edge pixels (gradient detection)
      if (px > x && py > y && px < x + w - 1 && py < y + h - 1) {
        const left = (py * imageWidth + (px - 1)) * 4
        const right = (py * imageWidth + (px + 1)) * 4
        const top = ((py - 1) * imageWidth + px) * 4
        const bottom = ((py + 1) * imageWidth + px) * 4
        
        const currentGray = gray
        const leftGray = (imageData[left] + imageData[left + 1] + imageData[left + 2]) / 3
        const rightGray = (imageData[right] + imageData[right + 1] + imageData[right + 2]) / 3
        const topGray = (imageData[top] + imageData[top + 1] + imageData[top + 2]) / 3
        const bottomGray = (imageData[bottom] + imageData[bottom + 1] + imageData[bottom + 2]) / 3
        
        const gradient = Math.abs(currentGray - leftGray) + Math.abs(currentGray - rightGray) + 
                        Math.abs(currentGray - topGray) + Math.abs(currentGray - bottomGray)
        
        if (gradient > params.edgeThreshold) {
          edgePixels++
        }
      }
    }
  }
  
  const darkDensity = (darkPixels / totalPixels) * 100
  const edgeDensity = (edgePixels / totalPixels) * 100
  
  // Enhanced detection criteria for whole tattoos
  const hasDarkContent = darkDensity > params.minDensity
  const hasEdges = edgeDensity > params.sensitivity / 10
  const hasEnoughContent = darkPixels > params.minArea / 100
  const hasSignificantComponent = maxComponentSize > params.minArea / 50
  // const hasReasonableComponents = connectedComponents > 0 && connectedComponents < 10 // Not too fragmented
  
  // More lenient detection for whole tattoo capture
  return hasDarkContent || hasEdges || hasEnoughContent || hasSignificantComponent
}

// Simple flood fill for connected component analysis
function floodFill(imageData: Uint8ClampedArray, visited: boolean[], startX: number, startY: number, 
                  offsetX: number, offsetY: number, w: number, h: number, imageWidth: number, threshold: number): number {
  const stack: [number, number][] = [[startX, startY]]
  let componentSize = 0
  
  while (stack.length > 0) {
    const [x, y] = stack.pop()!
    const localX = x - offsetX
    const localY = y - offsetY
    
    if (localX < 0 || localX >= w || localY < 0 || localY >= h) continue
    
    const localIndex = localY * w + localX
    if (visited[localIndex]) continue
    
    const i = (y * imageWidth + x) * 4
    const r = imageData[i]
    const g = imageData[i + 1]
    const b = imageData[i + 2]
    const gray = (r + g + b) / 3
    
    if (gray >= threshold) continue
    
    visited[localIndex] = true
    componentSize++
    
    // Add neighbors to stack
    stack.push([x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1])
  }
  
  return componentSize
}

// Merge overlapping/adjacent bounding boxes to form whole tattoos
function mergeOverlapping(
  boxes: Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number } }>
): Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number } }>
{
  if (boxes.length === 0) return boxes
  const sorted = [...boxes].sort((a, b) => a.bounds.x - b.bounds.x || a.bounds.y - b.bounds.y)
  const merged: typeof boxes = []
  const overlapPad = 8

  for (const b of sorted) {
    const last = merged[merged.length - 1]
    if (!last) {
      merged.push(b)
      continue
    }

    const a = last.bounds
    const c = b.bounds
    const ax2 = a.x + a.w
    const ay2 = a.y + a.h
    const cx2 = c.x + c.w
    const cy2 = c.y + c.h

    const overlaps = !(c.x > ax2 + overlapPad || cx2 < a.x - overlapPad || c.y > ay2 + overlapPad || cy2 < a.y - overlapPad)

    if (overlaps) {
      const nx = Math.min(a.x, c.x)
      const ny = Math.min(a.y, c.y)
      const nw = Math.max(ax2, cx2) - nx
      const nh = Math.max(ay2, cy2) - ny
      last.bounds = { x: nx, y: ny, w: nw, h: nh }
    } else {
      merged.push(b)
    }
  }

  // Re-id and names for merged
  return merged.map((m, idx) => ({
    ...m,
    id: `merged-${idx + 1}`,
    name: `item-${String(idx + 1).padStart(3, '0')}`
  }))
}

// Build polygon crops using contour → approx polygon → dilated mask workflow
async function recropWithPolygons(
  canvas: HTMLCanvasElement,
  _ctx: CanvasRenderingContext2D,
  processed: Uint8ClampedArray,
  boxes: Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number } }>,
  params: EdgeParams
): Promise<Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number; }, polygon: [number, number][] }>> {
  const result: Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number; }, polygon: [number, number][] }> = []

  for (const b of boxes) {
    const { x, y, w, h } = b.bounds

    // Extract ROI grayscale from processed data
    const roiGray = new Uint8ClampedArray(w * h)
    for (let ry = 0; ry < h; ry++) {
      for (let rx = 0; rx < w; rx++) {
        const gi = ((y + ry) * canvas.width + (x + rx)) * 4
        const r = processed[gi]
        const g = processed[gi + 1]
        const bcol = processed[gi + 2]
        roiGray[ry * w + rx] = (r + g + bcol) / 3
      }
    }

    // Threshold to binary mask (foreground=1)
    const bin = new Uint8Array(w * h)
    for (let i = 0; i < roiGray.length; i++) bin[i] = roiGray[i] < params.threshold ? 1 : 0

    // Dilate mask to create padding, then extract only EXTERNAL contour
    const radius = Math.max(2, Math.floor(params.padding))
    const paddedMask = dilateBinary(bin, w, h, radius)
    const paddedContour = traceExternalContour(paddedMask, w, h)
    if (paddedContour.length === 0) continue
    const epsilon = Math.max(1, Math.floor(params.approxEps * 100))
    const paddedApprox = simplifyPath(paddedContour, epsilon)

    // Create masked crop with alpha
    const off = new OffscreenCanvas(w, h)
    const offCtx = off.getContext('2d')!
    offCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h)
    offCtx.globalCompositeOperation = 'destination-in'
    offCtx.beginPath()
    paddedApprox.forEach(([px, py], idx) => {
      if (idx === 0) offCtx.moveTo(px, py)
      else offCtx.lineTo(px, py)
    })
    offCtx.closePath()
    offCtx.fill()
    offCtx.globalCompositeOperation = 'source-over'

    const blob = await off.convertToBlob({ type: 'image/png' })
    const polygonGlobal = paddedApprox.map(([px, py]) => [px + x, py + y] as [number, number])
    result.push({ id: b.id, blob, name: b.name, bounds: b.bounds, polygon: polygonGlobal })
  }

  return result
}

// Trace only the EXTERNAL contour (RETR_EXTERNAL equivalent)
function traceExternalContour(bin: Uint8Array, w: number, h: number): [number, number][] {
  // Border following: find the first foreground pixel and walk its border clockwise
  const dirs = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]] as const
  let startX = -1, startY = -1
  for (let y = 0; y < h && startX === -1; y++) {
    for (let x = 0; x < w; x++) {
      if (bin[y * w + x] === 1) { startX = x; startY = y; break }
    }
  }
  if (startX === -1) return []

  const contour: [number, number][] = []
  let cx = startX, cy = startY
  let prevDir = 0
  const visited = new Set<string>()
  for (let iter = 0; iter < w * h * 8; iter++) {
    contour.push([cx, cy])
    visited.add(`${cx},${cy}`)
    let found = false
    for (let k = 0; k < 8; k++) {
      const dir = (prevDir + k) % 8
      const nx = cx + dirs[dir][0]
      const ny = cy + dirs[dir][1]
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
      if (bin[ny * w + nx] === 1) {
        cx = nx; cy = ny; prevDir = (dir + 6) % 8; // bias turning right
        found = true
        break
      }
    }
    if (!found) break
    if (cx === startX && cy === startY && contour.length > 2) break
  }
  return contour
}

// Morphological dilation for binary images
function dilateBinary(src: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const dst = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let on = 0
      for (let dy = -radius; dy <= radius && !on; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          if (src[ny * w + nx] === 1) { on = 1; break }
        }
      }
      dst[y * w + x] = on
    }
  }
  return dst
}

// Morphological erosion for binary images
function erodeBinary(src: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  const dst = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let keep = 1
      for (let dy = -radius; dy <= radius && keep; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) { keep = 0; break }
          if (src[ny * w + nx] === 0) { keep = 0; break }
        }
      }
      dst[y * w + x] = keep
    }
  }
  return dst
}

// Circular (disk) morphology helpers to avoid blocky edges
function dilateBinaryDisk(src: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  if (radius <= 0) return src.slice()
  const r2 = radius * radius
  const dst = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let on = 0
      for (let dy = -radius; dy <= radius && !on; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > r2) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          if (src[ny * w + nx] === 1) { on = 1; break }
        }
      }
      dst[y * w + x] = on
    }
  }
  return dst
}

function erodeBinaryDisk(src: Uint8Array, w: number, h: number, radius: number): Uint8Array {
  if (radius <= 0) return src.slice()
  const r2 = radius * radius
  const dst = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let keep = 1
      for (let dy = -radius; dy <= radius && keep; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > r2) continue
          const nx = x + dx, ny = y + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) { keep = 0; break }
          if (src[ny * w + nx] === 0) { keep = 0; break }
        }
      }
      dst[y * w + x] = keep
    }
  }
  return dst
}

function morphOpenDisk(src: any, w: number, h: number, radius: number): Uint8Array {
  return dilateBinaryDisk(erodeBinaryDisk(src, w, h, radius), w, h, radius)
}

function morphCloseDisk(src: any, w: number, h: number, radius: number): Uint8Array {
  return erodeBinaryDisk(dilateBinaryDisk(src, w, h, radius), w, h, radius)
}

// RGB helpers: sRGB -> HSV and CIE Lab (approx)
function rgbToHsv(r: number, g: number, b: number): { h: number; s: number; v: number } {
  const rn = r / 255, gn = g / 255, bn = b / 255
  const max = Math.max(rn, gn, bn), min = Math.min(rn, gn, bn)
  const d = max - min
  let h = 0
  if (d !== 0) {
    switch (max) {
      case rn: h = ((gn - bn) / d + (gn < bn ? 6 : 0)); break
      case gn: h = (bn - rn) / d + 2; break
      default: h = (rn - gn) / d + 4
    }
    h /= 6
  }
  const s = max === 0 ? 0 : d / max
  const v = max
  return { h: h * 179, s: s * 255, v: v * 255 }
}

function srgbToLinear(c: number): number {
  const x = c / 255
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4)
}

function rgbToLab(r: number, g: number, b: number): { L: number; a: number; b: number } {
  // sRGB D65 → XYZ
  const R = srgbToLinear(r), G = srgbToLinear(g), B = srgbToLinear(b)
  let X = R * 0.4124564 + G * 0.3575761 + B * 0.1804375
  let Y = R * 0.2126729 + G * 0.7151522 + B * 0.0721750
  let Z = R * 0.0193339 + G * 0.1191920 + B * 0.9503041
  // Normalize by D65 white
  X /= 0.95047; Y /= 1.00000; Z /= 1.08883
  const f = (t: number) => t > 0.008856 ? Math.cbrt(t) : (7.787 * t + 16 / 116)
  const fx = f(X), fy = f(Y), fz = f(Z)
  const L = (116 * fy - 16) * (255 / 100) // scale L to [0..255]
  const a = (500 * (fx - fy)) * 1 + 128     // center around ~128
  const bb = (200 * (fy - fz)) * 1 + 128
  return { L: Math.max(0, Math.min(255, L)), a: Math.max(0, Math.min(255, a)), b: Math.max(0, Math.min(255, bb)) }
}

// K-means over features [L,a,b,S,V] on a downsampled grid
/* removed: kmeans (unused after dual-gate gray suppression)
function kmeans(features: number[][], K: number, iters = 8): { centers: number[][]; labels: number[] } {
  const N = features.length
  const D = features[0].length
  // init centers by random picks
  const centers: number[][] = []
  const used = new Set<number>()
  while (centers.length < K) {
    const idx = Math.floor(Math.random() * N)
    if (!used.has(idx)) { centers.push(features[idx].slice()); used.add(idx) }
  }
  const labels = new Array<number>(N).fill(0)
  for (let iter = 0; iter < iters; iter++) {
    // assign
    for (let i = 0; i < N; i++) {
      let best = 0, bestDist = Infinity
      const xi = features[i]
      for (let k = 0; k < K; k++) {
        let d = 0
        const ck = centers[k]
        for (let dIdx = 0; dIdx < D; dIdx++) { const t = xi[dIdx] - ck[dIdx]; d += t * t }
        if (d < bestDist) { bestDist = d; best = k }
      }
      labels[i] = best
    }
    // update
    const sums: number[][] = Array.from({ length: K }, () => Array(D).fill(0))
    const counts = new Array<number>(K).fill(0)
    for (let i = 0; i < N; i++) {
      const k = labels[i]
      counts[k]++
      const xi = features[i]
      for (let dIdx = 0; dIdx < D; dIdx++) sums[k][dIdx] += xi[dIdx]
    }
    for (let k = 0; k < K; k++) {
      if (counts[k] === 0) continue
      for (let dIdx = 0; dIdx < D; dIdx++) centers[k][dIdx] = sums[k][dIdx] / counts[k]
    }
  }
  return { centers, labels }
}
*/

// Compute adaptive gray center using downsampled k-means on LAB/HSV
/* removed: adaptive gray center (unused after dual-gate gray suppression)
export function computeAdaptiveGrayCenter(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  maxSide = 640,
  K = 4
): { center: { L: number; a: number; b: number; S: number; V: number }; frac: number } {
  const scale = Math.min(1, maxSide / Math.max(w, h))
  const sw = Math.max(1, Math.round(w * scale))
  const sh = Math.max(1, Math.round(h * scale))
  const features: number[][] = []
  const sampleEvery = Math.max(1, Math.floor(1 / scale)) // simple stride sampling from full-res
  for (let y = 0; y < h; y += sampleEvery) {
    for (let x = 0; x < w; x += sampleEvery) {
      const i = (y * w + x) * 4
      const r = data[i], g = data[i + 1], b = data[i + 2]
      const { h: _h, s, v } = rgbToHsv(r, g, b)
      const { L, a, b: bb } = rgbToLab(r, g, b)
      features.push([L, a, bb, s, v])
      if (features.length > sw * sh) break
    }
    if (features.length > sw * sh) break
  }
  const { centers, labels } = kmeans(features, K, 6)
  let best = -1, bestScore = -1, counts = new Array<number>(K).fill(0)
  for (const k of labels) counts[k]++
  for (let k = 0; k < K; k++) {
    const [Lc, ac, bc, Sc, Vc] = centers[k]
    const neutralA = Math.abs(ac - 128)
    const neutralB = Math.abs(bc - 128)
    const score = (255 - Sc) + 0.7 * Lc + 0.5 * Vc + (20 - Math.min(20, neutralA)) + (20 - Math.min(20, neutralB))
    if (score > bestScore) { bestScore = score; best = k }
  }
  const frac = counts[best] / labels.length
  const [Lc, ac, bc, Sc, Vc] = centers[best]
  return { center: { L: Lc, a: ac, b: bc, S: Sc, V: Vc }, frac }
}
*/

// Simple Sobel edge magnitude → binary mask (0/1)
function sobelEdgesBinary(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  threshold: number
): Uint8Array {
  const gray = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4
      gray[y * w + x] = (data[i] + data[i + 1] + data[i + 2]) / 3
    }
  }
  const out = new Uint8Array(w * h)
  const kx = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
  const ky = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      let gx = 0, gy = 0
      let idx = 0
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const v = gray[(y + dy) * w + (x + dx)]
          gx += v * kx[idx]
          gy += v * ky[idx]
          idx++
        }
      }
      const mag = Math.sqrt(gx * gx + gy * gy)
      out[y * w + x] = mag >= threshold ? 1 : 0
    }
  }
  return out
}

// Fill internal holes in a binary mask (foreground=1)
function fillHolesBinary(src: Uint8Array, w: number, h: number): Uint8Array {
  const outside = new Uint8Array(w * h)
  const stack: [number, number][] = []
  // seed from borders where src==0
  for (let x = 0; x < w; x++) {
    if (src[x] === 0) { outside[x] = 1; stack.push([x, 0]) }
    const by = (h - 1) * w + x
    if (src[by] === 0) { outside[by] = 1; stack.push([x, h - 1]) }
  }
  for (let y = 0; y < h; y++) {
    const lx = y * w
    if (src[lx] === 0) { outside[lx] = 1; stack.push([0, y]) }
    const rx = y * w + (w - 1)
    if (src[rx] === 0) { outside[rx] = 1; stack.push([w - 1, y]) }
  }
  const nbs = [[1,0],[-1,0],[0,1],[0,-1]] as const
  while (stack.length) {
    const [cx, cy] = stack.pop()!
    for (const [dx, dy] of nbs) {
      const nx = cx + dx, ny = cy + dy
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
      const idx = ny * w + nx
      if (src[idx] === 0 && outside[idx] === 0) { outside[idx] = 1; stack.push([nx, ny]) }
    }
  }
  const filled = new Uint8Array(src) // ensure ArrayBuffer type stays Uint8Array<ArrayBuffer>
  for (let i = 0; i < filled.length; i++) {
    if (src[i] === 0 && outside[i] === 0) filled[i] = 1
  }
  return filled
}

// Otsu threshold for grayscale [0..255]
function otsuThreshold(gray: Uint8Array): number {
  const hist = new Array(256).fill(0)
  for (let i = 0; i < gray.length; i++) hist[gray[i]]++
  const total = gray.length
  let sum = 0
  for (let t = 0; t < 256; t++) sum += t * hist[t]
  let sumB = 0
  let wB = 0
  let wF = 0
  let varMax = -1
  let threshold = 0
  for (let t = 0; t < 256; t++) {
    wB += hist[t]
    if (wB === 0) continue
    wF = total - wB
    if (wF === 0) break
    sumB += t * hist[t]
    const mB = sumB / wB
    const mF = (sum - sumB) / wF
    const between = wB * wF * (mB - mF) * (mB - mF)
    if (between > varMax) { varMax = between; threshold = t }
  }
  return threshold
}

// Build silhouette mask (ink=1) using grayscale + Otsu + morphology
function buildSilhouetteMask(
  data: Uint8ClampedArray,
  w: number,
  h: number,
  params: EdgeParams,
  _useGraySuppress = false
): Uint8Array {
  // A) Strict dark-ink silhouette (intensity-only)
  const gray = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4
      const r = data[i], g = data[i + 1], b = data[i + 2]
      gray[y * w + x] = Math.round((r + g + b) / 3)
    }
  }
  const otsu = otsuThreshold(gray)
  const T_DARK = Math.min(255, otsu + 12)
  let ink: any = new Uint8Array(w * h)
  for (let i = 0; i < gray.length; i++) ink[i] = gray[i] >= T_DARK ? 0 : 1
  // bridge tiny gaps (elliptical proxy via 3x3 close + 1x dilate)
  ink = morphCloseDisk(ink, w, h, 1) as unknown as Uint8Array
  ink = dilateBinaryDisk(ink, w, h, 1) as unknown as Uint8Array

  return ink
}

// Find all external contours from a binary mask
/* unused but kept for reference
function findExternalContours(bin: Uint8Array, w: number, h: number): Array<[number, number][]> {
  const contours: Array<[number, number][]> = []
  const visited = new Uint8Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      if (bin[idx] === 1 && !visited[idx]) {
        const contour = traceExternalContour(bin, w, h)
        // mark visited along contour
        for (const [cx, cy] of contour) visited[cy * w + cx] = 1
        if (contour.length > 3) contours.push(contour)
      }
    }
  }
  return contours
}
*/

// Compute polygon area (signed absolute)
/* unused but kept for reference
function polygonArea(points: [number, number][]): number {
  let area = 0
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i]
    const [xj, yj] = points[j]
    area += (xj + xi) * (yj - yi)
  }
  return Math.abs(area) / 2
}
*/

// Create crops directly from polygons (global coords)
async function cropsFromPolygons(
  canvas: HTMLCanvasElement,
  polygons: Array<{ id: string; polygon: [number, number][]; bounds: { x: number; y: number; w: number; h: number } }>
): Promise<Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number }; polygon: [number, number][] }>> {
  const out: Array<{ id: string; blob: Blob; name: string; bounds: { x: number; y: number; w: number; h: number }; polygon: [number, number][] }> = []
  let idx = 0
  for (const p of polygons) {
    const { x, y, w, h } = p.bounds
    const off = new OffscreenCanvas(w, h)
    const offCtx = off.getContext('2d')!
    offCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h)
    offCtx.globalCompositeOperation = 'destination-in'
    offCtx.beginPath()
    p.polygon.forEach(([px, py], i) => {
      const lx = px - x
      const ly = py - y
      if (i === 0) offCtx.moveTo(lx, ly)
      else offCtx.lineTo(lx, ly)
    })
    offCtx.closePath()
    offCtx.fill()
    offCtx.globalCompositeOperation = 'source-over'
    const blob = await off.convertToBlob({ type: 'image/png' })
    out.push({ id: `poly-${++idx}`, blob, name: `item-${String(idx).padStart(3, '0')}`, bounds: p.bounds, polygon: p.polygon })
  }
  return out
}

// Derive a 1px boundary from a filled mask using erosion subtraction
function boundaryFromMask(src: Uint8Array, w: number, h: number): Uint8Array {
  const er = erodeBinary(src, w, h, 1)
  const out = new Uint8Array(w * h)
  for (let i = 0; i < out.length; i++) out[i] = src[i] === 1 && er[i] === 0 ? 1 : 0
  return out
}

// Label connected components (foreground=1) with 4-neighborhood
function labelComponentsBinary(
  src: Uint8Array,
  w: number,
  h: number
): Array<{ pixels: [number, number][]; bounds: { x: number; y: number; w: number; h: number }; area: number }> {
  const visited = new Uint8Array(w * h)
  const components: Array<{ pixels: [number, number][]; bounds: { x: number; y: number; w: number; h: number }; area: number }> = []
  const stack: [number, number][] = []
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x
      if (src[idx] !== 1 || visited[idx] === 1) continue
      let minX = x, minY = y, maxX = x, maxY = y
      const pixels: [number, number][] = []
      stack.push([x, y])
      visited[idx] = 1
      while (stack.length) {
        const [cx, cy] = stack.pop()!
        pixels.push([cx, cy])
        if (cx < minX) minX = cx
        if (cy < minY) minY = cy
        if (cx > maxX) maxX = cx
        if (cy > maxY) maxY = cy
        const nbs = [[1,0],[-1,0],[0,1],[0,-1]] as const
        for (const [dx, dy] of nbs) {
          const nx = cx + dx, ny = cy + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          const nidx = ny * w + nx
          if (src[nidx] === 1 && visited[nidx] === 0) {
            visited[nidx] = 1
            stack.push([nx, ny])
          }
        }
      }
      components.push({ pixels, bounds: { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 }, area: pixels.length })
    }
  }
  return components
}

// Detect polygons using the silhouette pipeline and add uniform padding
async function detectPolygonsSilhouette(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  params: EdgeParams
): Promise<{ polygons: Array<{ id: string; polygon: [number, number][]; bounds: { x: number; y: number; w: number; h: number } }>; mask: { data: Uint8Array; width: number; height: number } }> {
  const { width, height } = canvas
  const rgba = ctx.getImageData(0, 0, width, height).data
  // In line-art mode, suppress gray labels from the silhouette
  const mask = buildSilhouetteMask(rgba, width, height, params, false)
  let comps = labelComponentsBinary(mask, width, height)

  // B) Cheap post-filter to drop gray plates using strict T_DARK
  const gray = new Uint8Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4
      const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2]
      gray[y * width + x] = Math.round((r + g + b) / 3)
    }
  }
  const otsu = otsuThreshold(gray)
  const T_DARK = Math.min(255, otsu + 12)
  const filtered: typeof comps = []
  for (const c of comps) {
    const r = c.bounds
    let darkCount = 0
    for (let yy = r.y; yy < r.y + r.h; yy++) {
      for (let xx = r.x; xx < r.x + r.w; xx++) {
        const gi = yy * width + xx
        if (gray[gi] < T_DARK) darkCount++
      }
    }
    const inkRatio = darkCount / (r.w * r.h)
    // mean brightness
    let sum = 0
    for (let yy = r.y; yy < r.y + r.h; yy++) {
      for (let xx = r.x; xx < r.x + r.w; xx++) {
        sum += gray[yy * width + xx]
      }
    }
    const meanL = sum / (r.w * r.h)
    const pass = !(inkRatio < 0.10 && meanL > 170)
    if (pass) filtered.push(c)
  }
  comps = filtered

  const totalPx = width * height
  // scale-normalized area (defaults ~0.12% of processed image)
  const minAreaPx = Math.max(Math.floor(0.0012 * totalPx), params.minContourArea)
  const maxAreaPx = Math.min(Math.floor(0.2 * totalPx), params.maxContourArea || totalPx)
  const pad = Math.max(12, Math.floor(params.padding))

  const polygons: Array<{ id: string; polygon: [number, number][]; bounds: { x: number; y: number; w: number; h: number } }> = []
  let idx = 0
  let droppedByArea = 0, droppedByAspect = 0, droppedByBorder = 0
  for (const c of comps) {
    if (c.area < minAreaPx || c.area > maxAreaPx) { droppedByArea++; continue }
    const touchesBorder = c.bounds.x <= 1 || c.bounds.y <= 1 || c.bounds.x + c.bounds.w >= width - 1 || c.bounds.y + c.bounds.h >= height - 1
    if (touchesBorder && c.area < 0.3 * totalPx) { droppedByBorder++; continue }

    // Safe ROI with margin so dilation never hits ROI border
    const { x, y, w, h } = c.bounds
    const aspect = w / h
    if (aspect < 0.2 || aspect > 5.0) { droppedByAspect++; continue }
    const margin = Math.max(pad + 6, 16)
    const x0 = Math.max(0, x - margin)
    const y0 = Math.max(0, y - margin)
    const x1 = Math.min(width, x + w + margin)
    const y1 = Math.min(height, y + h + margin)
    const rw = x1 - x0
    const rh = y1 - y0
    // per-item mask in safe ROI
    let item: any = new Uint8Array(rw * rh)
    for (const [px, py] of c.pixels) {
      const rx = px - x0
      const ry = py - y0
      if (rx >= 0 && ry >= 0 && rx < rw && ry < rh) item[ry * rw + rx] = 1
    }
    // hole fill to prevent inner boxes
    item = fillHolesBinary(item, rw, rh)
    // gentle smoothing with circular kernels
    item = morphOpenDisk(item, rw, rh, 2)
    item = morphCloseDisk(item, rw, rh, 3)
    // outward-only padding (per item) with circular kernel
    item = dilateBinaryDisk(item, rw, rh, pad)

    // Trace polygon on boundary and simplify with gentle epsilon
    const boundary = boundaryFromMask(item, rw, rh)
    const polyLocal = traceExternalContour(boundary, rw, rh)
    if (polyLocal.length < 3) continue
    const periApprox = Math.max(w, h) * 2 // fallback scale
    const simplified = simplifyPath(polyLocal, Math.max(2, Math.floor(0.008 * periApprox)))
    const polyGlobal = simplified.map(([px, py]) => [px + x0, py + y0] as [number, number])
    // compute bounds from polygon to keep consistent with padding margin
    let minX = width, minY = height, maxX = 0, maxY = 0
    for (const [gx, gy] of polyGlobal) {
      if (gx < minX) minX = gx
      if (gy < minY) minY = gy
      if (gx > maxX) maxX = gx
      if (gy > maxY) maxY = gy
    }
    polygons.push({ id: `tat-${++idx}`, polygon: polyGlobal, bounds: { x: Math.max(0, Math.floor(minX)), y: Math.max(0, Math.floor(minY)), w: Math.min(width - Math.floor(minX), Math.ceil(maxX - minX)), h: Math.min(height - Math.floor(minY), Math.ceil(maxY - minY)) } })
  }

  // Pass B: thin/small salvage using edges → regions
  const edges = sobelEdgesBinary(rgba, width, height, 60)
  // thicken and close small gaps
  let edgesThick = dilateBinaryDisk(edges, width, height, 1)
  edgesThick = morphCloseDisk(edgesThick, width, height, 2)
  // convert to regions by closing and opening lightly
  let thinRegions = morphCloseDisk(edgesThick, width, height, 2)
  thinRegions = morphOpenDisk(thinRegions, width, height, 1)
  const compsB = labelComponentsBinary(thinRegions, width, height)
  const minAreaPxB = Math.max(1, Math.floor(0.00015 * totalPx))
  const maxAreaPxB = Math.max(minAreaPxB + 1, Math.floor(0.01 * totalPx))
  for (const c of compsB) {
    if (c.area < minAreaPxB || c.area > maxAreaPxB) continue
    const { x, y, w, h } = c.bounds
    const aspect = w / h
    if (aspect < 0.1 || aspect > 8.0) continue
    // thinness guard: perimeter/area proxy using boundary pixels
    const boundary = boundaryFromMask((() => { const m = new Uint8Array(w * h); for (const [px, py] of c.pixels) m[(py - y) * w + (px - x)] = 1; return m })(), w, h)
    let perim = 0
    for (let i = 0; i < boundary.length; i++) if (boundary[i] === 1) perim++
    if (perim / Math.max(c.area, 1) < 0.12) continue
    // build padded polygon with safe ROI
    const margin = Math.max(pad + 6, 16)
    const x0 = Math.max(0, x - margin)
    const y0 = Math.max(0, y - margin)
    const x1 = Math.min(width, x + w + margin)
    const y1 = Math.min(height, y + h + margin)
    const rw = x1 - x0, rh = y1 - y0
    let item: any = new Uint8Array(rw * rh)
    for (const [px, py] of c.pixels) {
      const rx = px - x0, ry = py - y0
      if (rx >= 0 && ry >= 0 && rx < rw && ry < rh) item[ry * rw + rx] = 1
    }
    item = fillHolesBinary(item, rw, rh) as unknown as Uint8Array
    item = morphCloseDisk(item, rw, rh, 2) as unknown as Uint8Array
    item = dilateBinaryDisk(item, rw, rh, pad) as unknown as Uint8Array
    item = fillHolesBinary(item, rw, rh) as unknown as Uint8Array
    item = morphCloseDisk(item, rw, rh, 2) as unknown as Uint8Array
    item = dilateBinaryDisk(item, rw, rh, pad) as unknown as Uint8Array
    const b2 = boundaryFromMask(item, rw, rh)
    const polyLocal = traceExternalContour(b2, rw, rh)
    if (polyLocal.length < 3) continue
    const periApprox = Math.max(w, h) * 2
    const simplified = simplifyPath(polyLocal, Math.max(1, Math.floor(0.006 * periApprox)))
    const polyGlobal = simplified.map(([px, py]) => [px + x0, py + y0] as [number, number])
    // de-duplicate with existing by IoU
    let dup = false
    for (const p of polygons) {
      const iou = polygonIoU(polygonToBBox(p.polygon), polygonToBBox(polyGlobal))
      if (iou > 0.6) { dup = true; break }
    }
    if (!dup) {
      let minX = width, minY = height, maxX = 0, maxY = 0
      for (const [gx, gy] of polyGlobal) { if (gx < minX) minX = gx; if (gy < minY) minY = gy; if (gx > maxX) maxX = gx; if (gy > maxY) maxY = gy }
      polygons.push({ id: `tat-${++idx}`, polygon: polyGlobal, bounds: { x: Math.floor(minX), y: Math.floor(minY), w: Math.ceil(maxX - minX), h: Math.ceil(maxY - minY) } })
    }
  }

  console.table({ totalComponents: comps.length, kept: polygons.length, droppedByArea, droppedByAspect, droppedByBorder })
  return { polygons, mask: { data: mask, width, height } }
}

function polygonToBBox(poly: [number, number][]): { x: number; y: number; w: number; h: number } {
  let minX = Number.POSITIVE_INFINITY, minY = Number.POSITIVE_INFINITY, maxX = 0, maxY = 0
  for (const [x, y] of poly) { if (x < minX) minX = x; if (y < minY) minY = y; if (x > maxX) maxX = x; if (y > maxY) maxY = y }
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY }
}

function bboxIoU(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }): number {
  const ax2 = a.x + a.w, ay2 = a.y + a.h, bx2 = b.x + b.w, by2 = b.y + b.h
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x))
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y))
  const inter = ix * iy
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

function polygonIoU(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }): number {
  return bboxIoU(a, b)
}

// ---------- Inner-island suppression (no external libs) ----------
function polygonAreaSigned(poly: [number, number][]): number {
  let s = 0
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i]
    const [x2, y2] = poly[(i + 1) % poly.length]
    s += x1 * y2 - x2 * y1
  }
  return s / 2
}

function pointInPolygon(x: number, y: number, poly: [number, number][]): boolean {
  // Ray casting
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1]
    const xj = poly[j][0], yj = poly[j][1]
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi
    if (intersect) inside = !inside
  }
  return inside
}

function fractionVerticesInside(inner: [number, number][], outer: [number, number][]): number {
  if (inner.length === 0) return 0
  let insideCount = 0
  for (const [x, y] of inner) if (pointInPolygon(x, y, outer)) insideCount++
  return insideCount / inner.length
}

function suppressInnerIslands(
  polygons: Array<{ id: string; polygon: [number, number][]; bounds: { x: number; y: number; w: number; h: number } }>,
  containmentTol = 0.98
): Array<{ id: string; polygon: [number, number][]; bounds: { x: number; y: number; w: number; h: number } }> {
  const withArea = polygons.map(p => ({...p, area: Math.abs(polygonAreaSigned(p.polygon)) }))
  const sorted = [...withArea].sort((a, b) => b.area - a.area)
  const keep: typeof withArea = []
  for (const q of sorted) {
    let contained = false
    for (const p of keep) {
      // fast bbox rejection
      const qb = polygonToBBox(q.polygon)
      const pb = polygonToBBox(p.polygon)
      const bboxInside = qb.x >= pb.x && qb.y >= pb.y && qb.x + qb.w <= pb.x + pb.w && qb.y + qb.h <= pb.y + pb.h
      if (!bboxInside) continue
      const frac = fractionVerticesInside(q.polygon, p.polygon)
      if (frac >= containmentTol) { contained = true; break }
    }
    if (!contained) keep.push(q)
  }
  return keep.map(({area, ...rest}) => rest)
}

// Simplify path via radial-distance-like reduction (proxy for Douglas-Peucker for speed)
function simplifyPath(points: [number, number][], tolerance: number): [number, number][] {
  if (points.length <= 3) return points
  const simplified: [number, number][] = []
  let last = points[0]
  simplified.push(last)
  for (let i = 1; i < points.length; i++) {
    const [x, y] = points[i]
    const dx = x - last[0]
    const dy = y - last[1]
    if (dx * dx + dy * dy >= tolerance * tolerance) {
      simplified.push([x, y])
      last = [x, y]
    }
  }
  if (simplified[simplified.length - 1] !== points[points.length - 1]) simplified.push(points[points.length - 1])
  return simplified
}

// Very lightweight polygon dilation: paint polygon to mask and expand by N using square kernel
/* unused but kept for reference
function dilatePolygonMask(poly: [number, number][], w: number, h: number, radius: number): Uint8Array {
  const mask = new Uint8Array(w * h)

  // Rasterize filled polygon into mask using winding rule approximation
  // Draw horizontal scanlines between minX/maxX for each y
  const edges: Array<[number, number, number]> = [] // [y, xStart, xEnd]
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i]
    const [x2, y2] = poly[(i + 1) % poly.length]
    const minY = Math.min(y1, y2)
    const maxY = Math.max(y1, y2)
    for (let yv = Math.ceil(minY); yv <= Math.floor(maxY); yv++) {
      const t = (yv - y1) / (y2 - y1 || 1)
      const xv = x1 + t * (x2 - x1)
      edges.push([yv, xv, xv])
    }
  }
  // Fill naive: set pixels near polygon vertices/edges
  for (const [py, px] of poly) {
    const ix = Math.round(px), iy = Math.round(py)
    if (ix >= 0 && iy >= 0 && ix < w && iy < h) mask[iy * w + ix] = 1
  }

  // Dilate by square kernel radius
  if (radius <= 0) return mask
  const out = new Uint8Array(w * h)
  for (let yv = 0; yv < h; yv++) {
    for (let xv = 0; xv < w; xv++) {
      let on = 0
      for (let dy = -radius; dy <= radius && !on; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = xv + dx, ny = yv + dy
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue
          if (mask[ny * w + nx] === 1) { on = 1; break }
        }
      }
      out[yv * w + xv] = on
    }
  }
  return out
}
*/

interface EdgeParams {
  blurKernel: number
  cannyLow: number
  cannyHigh: number
  minArea: number
  padding: number
  morphClose: number
  morphDilate: number
  threshold: number
  adaptiveBlock: number
  adaptiveC: number
  approxEps: number
  minContourArea: number
  maxContourArea: number
  aspectRatioMin: number
  aspectRatioMax: number
  // New controls
  gridSize: number
  sensitivity: number
  minDensity: number
  edgeThreshold: number
  noiseReduction: number
  contrastBoost: number
  brightness: number
  saturation: number
  hueShift: number
  gamma: number
  sharpen: number
  erosion: number
  dilation: number
  opening: number
  closing: number
}

// Inline help copy for controls
const HELP = {
  preset: {
    short: 'Loads tuned defaults for a workflow',
    help: 'Switch presets to load recommended values. After tweaking, the preset remembers your last values until you pick another preset.'
  },
  blur: {
    short: 'Light denoise before edges',
    help: 'Small blur to smooth pixels. Higher = fewer specks, softer lines. Start 0–2.'
  },
  cannyLow: {
    short: 'Lower edge threshold',
    help: 'Canny hysteresis: Lower → more edges; raise to remove fuzz. Keep High about 2–3× Low.'
  },
  cannyHigh: {
    short: 'Upper edge threshold',
    help: 'Canny upper bound. Higher = stricter (fewer) edges. Keep ~2–3× Low.'
  },
  threshold: {
    short: 'Binarize ink vs paper',
    help: 'Global threshold used in silhouette. Higher = fewer ink pixels. Often used with Otsu/adaptive.'
  },
  minArea: {
    short: 'Smallest object to keep',
    help: 'Drops tiny contours like dust/numbers. Units are processed pixels and auto-scale.'
  },
  padding: {
    short: 'Gap between art & red line',
    help: 'Outward dilation in pixels for the loose boundary. Larger = more whitespace.'
  },
  morphClose: {
    short: 'Bridge tiny gaps',
    help: 'Close = dilate then erode. Higher merges nearby strokes; too high may join neighbors.'
  },
  morphDilate: {
    short: 'Thicken/merge strokes',
    help: 'Extra dilation iterations. Helps unify broken lines; too high may glue adjacent tattoos.'
  },
  adaptiveBlock: {
    short: 'Local window (odd px)',
    help: 'Neighborhood size for adaptive threshold. Bigger window = smoother, less shadow-sensitive. Use odd numbers 9–21.'
  },
  adaptiveC: {
    short: 'Bias for adaptive',
    help: 'Constant subtracted from local mean. Higher C = stricter; lower C = more inclusive.'
  },
  approxEps: {
    short: 'Polygon smoothness',
    help: 'Douglas–Peucker epsilon (fraction of perimeter). Lower = truer shape; higher = simpler polygons. Try 0.008–0.015.'
  },
  minContour: {
    short: 'Min contour size gate',
    help: 'Filters contours with very small perimeter/area. Increase to ignore tiny fragments.'
  },
  maxContour: {
    short: 'Max contour size gate',
    help: 'Filters extremely large blobs (e.g., page border). Lower if a frame is picked up.'
  },
  aspectMin: {
    short: 'Min width/height ratio',
    help: 'Rejects super-skinny shapes (e.g., stray lines). 0.2 means width can be 1/5 of height.'
  },
  aspectMax: {
    short: 'Max width/height ratio',
    help: 'Rejects super-wide shapes. 5.0 means width up to 5× height.'
  },
  gridSize: {
    short: 'Processing grid granularity',
    help: 'Internal tiling/merging granularity. Smaller = finer detail, more CPU. Larger = faster, coarser.'
  },
  sensitivity: {
    short: 'Detector aggressiveness',
    help: 'Master gain for how eager detection is. Higher = more picks (and more noise). Lower = stricter.'
  },
  minDensity: {
    short: 'Minimum dark-pixel density',
    help: 'Grid cell must exceed this dark pixel percentage to be considered.'
  },
  edgeThreshold: {
    short: 'Edge gradient threshold',
    help: 'Higher = require stronger gradients to count as an edge within a grid cell.'
  },
  brightness: { short: 'Add/subtract to RGB', help: 'Adjusts brightness pre-processing. Left = darker; right = brighter.' },
  contrastBoost: { short: 'Expand contrast around midtones', help: 'Left = lower contrast; right = higher contrast which may increase edges.' },
  gamma: { short: 'Gamma correction', help: 'Left (<1) brightens shadows; right (>1) darkens shadows. Affects silhouette.' },
  sharpen: { short: 'Unsharp-like boost', help: 'Higher slightly sharpens before detection, but may add noise.' },
  erosion: { short: 'Extra erosion', help: 'Removes small specks/bridges. Too high thins art.' },
  dilation: { short: 'Extra dilation', help: 'Thickens art/bridges gaps. Too high can merge neighbors.' },
  opening: { short: 'Open (erode→dilate)', help: 'Removes tiny noise while preserving shape.' },
  closing: { short: 'Close (dilate→erode)', help: 'Fills tiny holes/gaps without expanding outward too far.' },
} as const

type SliderRowProps = {
  id: string
  label: string
  short?: string
  help?: string
  min: number | string
  max: number | string
  step?: number | string
  value: number
  disabled?: boolean
  onChange: (v: number) => void
}

function SliderRow({ id, label, short, help, min, max, step, value, disabled, onChange }: SliderRowProps) {
  return (
    <div>
      <LabelWithHelp id={id} label={label} short={short} help={help} />
      <div className="flex items-center gap-2">
        <input
          id={id}
          aria-describedby={`${id}-help`}
          type="range"
          min={min as number}
          max={max as number}
          step={step as number | undefined}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full h-1 disabled:opacity-50"
        />
        <span className="text-[11px] text-neutral-400 px-1.5 py-0.5 rounded bg-neutral-800 min-w-[34px] text-right">
          {typeof value === 'number' ? value : String(value)}
        </span>
      </div>
    </div>
  )
}

function App() {
  const [image, setImage] = useState<string | null>(null)
  const [crops, setCrops] = useState<Array<{id: string, blob: Blob, name: string, bounds: {x: number, y: number, w: number, h: number}, polygon?: [number, number][], detectedText?: string, ocrConf?: number, needsReview?: boolean}>>([])
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameInput, setRenameInput] = useState<string>('')
  
  async function recognizeTextFromRoi(roiCanvas: HTMLCanvasElement): Promise<{ text: string | null; conf: number }> {
    const scale = 2
    const off = document.createElement('canvas')
    off.width = roiCanvas.width * scale
    off.height = roiCanvas.height * scale
    const octx = off.getContext('2d')!
    octx.fillStyle = 'white'
    octx.fillRect(0, 0, off.width, off.height)
    octx.imageSmoothingEnabled = false
    octx.drawImage(roiCanvas, 0, 0, off.width, off.height)
    const res7 = await Tesseract.recognize(off, 'eng', { logger: () => {} })
    let raw = (res7.data.text || '').trim()
    let conf = res7.data.confidence || 0
    if (!raw) {
      const res6 = await Tesseract.recognize(off, 'eng', { logger: () => {} })
      raw = (res6.data.text || '').trim()
      conf = Math.max(conf, res6.data.confidence || 0)
    }
    // whitelist post-filter
    raw = raw.replace(/[^0-9-]/g, '')
    const ok = /^([0-9]{3,})(-[0-9]{2})?$/.test(raw)
    return { text: ok ? raw : null, conf }
  }

  async function detectAndNameAll() {
    if (!canvasRef.current || crops.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d', { willReadFrequently: true })!
    const img = imgRef.current
    if (!img) return
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    ctx.drawImage(img, 0, 0)
    const updated: typeof crops = []
    for (const c of crops) {
      if (!c.polygon) { updated.push(c); continue }
      const ringPx = 40
      const bb = c.bounds
      const rx = Math.max(0, bb.x - ringPx)
      const ry = Math.max(0, bb.y - ringPx)
      const rw = Math.min(canvas.width - rx, bb.w + ringPx * 2)
      const rh = Math.min(canvas.height - ry, bb.h + ringPx * 2)
      if (Math.hypot(rw - bb.w, rh - bb.h) > 100) {
        updated.push({ ...c, needsReview: true })
        continue
      }
      const roi = document.createElement('canvas')
      roi.width = rw
      roi.height = rh
      const rctx = roi.getContext('2d')!
      rctx.fillStyle = 'white'
      rctx.fillRect(0, 0, rw, rh)
      rctx.drawImage(canvas, rx, ry, rw, rh, 0, 0, rw, rh)
      const { text, conf } = await recognizeTextFromRoi(roi)
      if (text && conf >= 70) {
        updated.push({ ...c, detectedText: text, name: text, ocrConf: conf, needsReview: false })
      } else {
        updated.push({ ...c, detectedText: text ?? undefined, ocrConf: conf, needsReview: true })
      }
    }
    setCrops(updated)
  }
  const [isProcessing, setIsProcessing] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState<string>('line-art')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const [naturalSize, setNaturalSize] = useState<{w: number; h: number}>({ w: 1, h: 1 })
  const [displaySize, setDisplaySize] = useState<{w: number; h: number}>({ w: 1, h: 1 })

  // Enhanced parameters for better detection
  const [params, setParams] = useState<EdgeParams>({
    blurKernel: 1,
    cannyLow: 30,
    cannyHigh: 100,
    minArea: 500,
    padding: 8,
    morphClose: 2,
    morphDilate: 1,
    threshold: 128,
    adaptiveBlock: 11,
    adaptiveC: 2,
    approxEps: 0.02,
    minContourArea: 1000,
    maxContourArea: 50000,
    aspectRatioMin: 0.3,
    aspectRatioMax: 3.0,
    // New controls with better defaults
    gridSize: 150,
    sensitivity: 50,
    minDensity: 10,
    edgeThreshold: 100,
    noiseReduction: 0,
    contrastBoost: 0,
    brightness: 0,
    saturation: 0,
    hueShift: 0,
    gamma: 1.0,
    sharpen: 0,
    erosion: 0,
    dilation: 0,
    opening: 0,
    closing: 0
  })

  const displayScale = naturalSize.w ? displaySize.w / naturalSize.w : 1

  const updateParam = (key: keyof EdgeParams, value: number) => setParams({ ...params, [key]: value })

  // Preset configurations for different tattoo types
  const presets = {
    'line-art': {
      name: 'Line Art Tattoos',
      params: {
        blurKernel: 1, cannyLow: 20, cannyHigh: 80, minArea: 600,
        // tuned for silhouette pipeline
        padding: 18, morphClose: 2, morphDilate: 1, threshold: 100,
        adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
        minContourArea: 500, maxContourArea: 100000,
        aspectRatioMin: 0.2, aspectRatioMax: 5.0,
        gridSize: 120, sensitivity: 30, minDensity: 0.6, edgeThreshold: 50,
        noiseReduction: 0, contrastBoost: 20, brightness: 10, saturation: 0,
        hueShift: 0, gamma: 1.2, sharpen: 2, erosion: 0, dilation: 0,
        opening: 0, closing: 1
      }
    },
    'detailed-art': {
      name: 'Detailed Tattoos',
      params: {
        blurKernel: 1, cannyLow: 15, cannyHigh: 60, minArea: 500,
        padding: 15, morphClose: 2, morphDilate: 1, threshold: 90,
        adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
        minContourArea: 800, maxContourArea: 150000,
        aspectRatioMin: 0.3, aspectRatioMax: 4.0,
        gridSize: 100, sensitivity: 40, minDensity: 8, edgeThreshold: 40,
        noiseReduction: 0, contrastBoost: 30, brightness: 15, saturation: 0,
        hueShift: 0, gamma: 1.3, sharpen: 3, erosion: 0, dilation: 1,
        opening: 0, closing: 2
      }
    },
    'simple-designs': {
      name: 'Simple Designs',
      params: {
        blurKernel: 1, cannyLow: 25, cannyHigh: 100, minArea: 150,
        padding: 10, morphClose: 1, morphDilate: 0, threshold: 110,
        adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
        minContourArea: 300, maxContourArea: 80000,
        aspectRatioMin: 0.2, aspectRatioMax: 6.0,
        gridSize: 140, sensitivity: 25, minDensity: 3, edgeThreshold: 60,
        noiseReduction: 0, contrastBoost: 15, brightness: 5, saturation: 0,
        hueShift: 0, gamma: 1.1, sharpen: 1, erosion: 0, dilation: 0,
        opening: 0, closing: 0
      }
    },
    'high-contrast': {
      name: 'High Contrast',
      params: {
        blurKernel: 1, cannyLow: 10, cannyHigh: 50, minArea: 100,
        padding: 8, morphClose: 0, morphDilate: 0, threshold: 80,
        adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
        minContourArea: 200, maxContourArea: 200000,
        aspectRatioMin: 0.1, aspectRatioMax: 10.0,
        gridSize: 80, sensitivity: 60, minDensity: 2, edgeThreshold: 30,
        noiseReduction: 0, contrastBoost: 50, brightness: 20, saturation: 0,
        hueShift: 0, gamma: 1.5, sharpen: 5, erosion: 0, dilation: 0,
        opening: 0, closing: 0
      }
    },
    'low-contrast': {
      name: 'Low Contrast',
      params: {
        blurKernel: 1, cannyLow: 5, cannyHigh: 30, minArea: 50,
        padding: 5, morphClose: 0, morphDilate: 0, threshold: 60,
        adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
        minContourArea: 100, maxContourArea: 300000,
        aspectRatioMin: 0.1, aspectRatioMax: 15.0,
        gridSize: 60, sensitivity: 80, minDensity: 1, edgeThreshold: 20,
        noiseReduction: 0, contrastBoost: 80, brightness: 30, saturation: 0,
        hueShift: 0, gamma: 2.0, sharpen: 8, erosion: 0, dilation: 0,
        opening: 0, closing: 0
      }
    }
  }

  const applyPreset = (presetKey: string) => {
    setSelectedPreset(presetKey)
  }

  // persist per-preset values
  // load values when preset changes (runs after initial params state)
  useEffect(() => {
    if (selectedPreset === 'custom') return
    const key = `params:${selectedPreset}`
    const raw = localStorage.getItem(key)
    if (raw) {
      try { setParams(JSON.parse(raw)) } catch { setParams(presets[selectedPreset as keyof typeof presets].params) }
    } else {
      setParams(presets[selectedPreset as keyof typeof presets].params)
    }
  }, [selectedPreset])

  useEffect(() => {
    if (selectedPreset === 'custom') return
    const key = `params:${selectedPreset}`
    try { localStorage.setItem(key, JSON.stringify(params)) } catch {}
  }, [params, selectedPreset])
  

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    const url = URL.createObjectURL(file)
    setImage(url)
    setCrops([])
  }

  // Simple and fast detection using grid-based approach
  const processImage = async () => {
    if (!image || !canvasRef.current) return
    
    setIsProcessing(true)
    
    try {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!
      const img = new Image()
      
      img.onload = async () => {
        canvas.width = img.width
        canvas.height = img.height
        ctx.drawImage(img, 0, 0)
        
        // If silhouette-friendly preset, run the silhouette pipeline on full image
        if (selectedPreset === 'line-art' || selectedPreset === 'detailed-art') {
          try {
            const { polygons: polys } = await detectPolygonsSilhouette(canvas, ctx, params)
            const filtered = suppressInnerIslands(polys, 0.98)
            const polygonCrops = await cropsFromPolygons(canvas, filtered)
            setCrops(polygonCrops)
          } finally {
            setIsProcessing(false)
          }
          return
        }

        // Fallback: grid-based detection (legacy)
        const newCrops: Array<{id: string, blob: Blob, name: string, bounds: {x: number, y: number, w: number, h: number}}> = []
        const processedImageData = preprocessImage(ctx, canvas.width, canvas.height, params)
        const baseGridSize = params.gridSize
        const gridSize = baseGridSize + params.padding * 2
        const cols = Math.floor(canvas.width / gridSize)
        const rows = Math.floor(canvas.height / gridSize)
        const overlap = Math.floor(baseGridSize * 0.1)
        let processedCount = 0
        const totalCells = rows * cols
        for (let row = 0; row < rows; row++) {
          for (let col = 0; col < cols; col++) {
            const x = Math.max(0, col * gridSize + params.padding - (col > 0 ? overlap : 0))
            const y = Math.max(0, row * gridSize + params.padding - (row > 0 ? overlap : 0))
            const w = Math.min(canvas.width - x, gridSize - params.padding * 2 + (col > 0 ? overlap : 0) + (col < cols - 1 ? overlap : 0))
            const h = Math.min(canvas.height - y, gridSize - params.padding * 2 + (row > 0 ? overlap : 0) + (row < rows - 1 ? overlap : 0))
            if (x + w < canvas.width && y + h < canvas.height) {
              const hasContent = detectContent(processedImageData, x, y, w, h, canvas.width, params)
              if (hasContent) {
                const cropCanvas = document.createElement('canvas')
                const cropCtx = cropCanvas.getContext('2d')!
                cropCanvas.width = w
                cropCanvas.height = h
                cropCtx.fillStyle = 'white'
                cropCtx.fillRect(0, 0, w, h)
                cropCtx.drawImage(canvas, x, y, w, h, 0, 0, w, h)
                cropCanvas.toBlob((blob) => {
                  if (blob) {
                    newCrops.push({ id: `crop-${row}-${col}`, blob, name: `item-${String(newCrops.length + 1).padStart(3, '0')}`, bounds: { x, y, w, h } })
                  }
                  processedCount++
                  if (processedCount === totalCells) {
                    const merged = mergeOverlapping(newCrops)
                    recropWithPolygons(canvas, ctx, processedImageData, merged, params).then((polygonCrops) => {
                      setCrops(polygonCrops)
                    })
                    setIsProcessing(false)
                  }
                })
              } else {
                processedCount++
                if (processedCount === totalCells) {
                  const merged = mergeOverlapping(newCrops)
                  recropWithPolygons(canvas, ctx, processedImageData, merged, params).then((polygonCrops) => {
                    setCrops(polygonCrops)
                  })
                  setIsProcessing(false)
                }
              }
            } else {
              processedCount++
              if (processedCount === totalCells) {
                const merged = mergeOverlapping(newCrops)
                recropWithPolygons(canvas, ctx, processedImageData, merged, params).then((polygonCrops) => {
                  setCrops(polygonCrops)
                })
                setIsProcessing(false)
              }
            }
          }
        }
        if (newCrops.length === 0) setIsProcessing(false)
      }
      
      img.src = image
    } catch (error) {
      console.error('Processing error:', error)
      setIsProcessing(false)
    }
  }

  const downloadCrop = (crop: {id: string, blob: Blob, name: string}) => {
    const url = URL.createObjectURL(crop.blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${crop.name}.png`
    a.click()
    URL.revokeObjectURL(url)
  }

  const downloadAll = () => {
    const nameCounts: Record<string, number> = {}
    const padded = (n: number) => String(n).padStart(4, '0')
    crops.forEach((crop, index) => {
      const base = crop.name
      nameCounts[base] = (nameCounts[base] || 0) + 1
      const suffix = nameCounts[base] > 1 ? `-${nameCounts[base]}` : ''
      const finalName = `${padded(index + 1)}-${base}${suffix}`
      const url = URL.createObjectURL(crop.blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `${finalName}.png`
      setTimeout(() => { a.click(); URL.revokeObjectURL(url) }, index * 100)
    })
    // Also download manifest.json
    const manifest = crops.map((c, idx) => ({
      id: c.id,
      name: c.name,
      detectedText: c.detectedText ?? null,
      confidence: c.ocrConf ?? null,
      needsReview: !!c.needsReview,
      polygon: c.polygon ?? null,
      bbox: c.bounds,
      order: idx + 1
    }))
    const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'manifest.json'
    setTimeout(() => { a.click(); URL.revokeObjectURL(url) }, crops.length * 120)
  }

  return (
    <div className="h-screen w-full flex flex-col bg-neutral-950 text-white">
      {/* Header */}
      <div className="h-12 bg-neutral-900 border-b border-neutral-800 flex items-center px-4 gap-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileUpload}
          className="hidden"
          id="file-input"
        />
        <label
          htmlFor="file-input"
          className="px-3 py-1 rounded bg-sky-700 hover:bg-sky-600 cursor-pointer"
        >
          Upload Image
        </label>
        
        <button
          onClick={processImage}
          disabled={!image || isProcessing}
          className="px-3 py-1 rounded bg-emerald-700 hover:bg-emerald-600 disabled:opacity-50"
        >
          {isProcessing ? 'Processing...' : 'Detect & Crop'}
        </button>
        
        {crops.length > 0 && (
          <button
            onClick={downloadAll}
            className="px-3 py-1 rounded bg-blue-700 hover:bg-blue-500"
          >
            Download All ({crops.length})
          </button>
        )}
        {crops.length > 0 && (
          <button
            onClick={async () => {
              await detectAndNameAll()
            }}
            className="px-3 py-1 rounded bg-purple-700 hover:bg-purple-600"
          >
            Detect text & name
          </button>
        )}
        
        <div className="text-sm text-neutral-400">
          {crops.length} crops ready
        </div>
      </div>

      {/* Main Content - 10%, 45%, 45% */}
      <div className="flex-1 flex gap-2 p-2">
        {/* Edge Controls - 10% */}
        <div className="w-[10%] bg-neutral-900 rounded border border-neutral-800 p-2 overflow-auto">
          <h3 className="text-sm font-semibold mb-3">Edge Controls</h3>
          
          {/* Preset Dropdown */}
          <div className="mb-4">
            <LabelWithHelp id="preset" label="Preset" short={HELP.preset.short} help={HELP.preset.help} />
            <select
              value={selectedPreset}
              onChange={(e) => applyPreset(e.target.value)}
              className="w-full text-xs bg-neutral-800 border border-neutral-700 rounded px-2 py-1"
            >
              <option value="custom">Custom</option>
              <option value="line-art">Line Art Tattoos</option>
              <option value="detailed-art">Detailed Tattoos</option>
              <option value="simple-designs">Simple Designs</option>
              <option value="high-contrast">High Contrast</option>
              <option value="low-contrast">Low Contrast</option>
            </select>
          </div>
          
          <div className="space-y-3 text-xs">
            
            <SliderRow id="blur" label="Blur" short={HELP.blur.short} help={HELP.blur.help} min={1} max={11} step={2} value={params.blurKernel} onChange={(v)=>updateParam('blurKernel', v)} />

            <SliderRow id="cannyLow" label="Canny Low" short={HELP.cannyLow.short} help={HELP.cannyLow.help} min={10} max={100} value={params.cannyLow} onChange={(v)=>updateParam('cannyLow', v)} />

            <SliderRow id="cannyHigh" label="Canny High" short={HELP.cannyHigh.short} help={HELP.cannyHigh.help} min={50} max={300} value={params.cannyHigh} onChange={(v)=>updateParam('cannyHigh', v)} />

            <SliderRow id="threshold" label="Threshold" short={HELP.threshold.short} help={HELP.threshold.help} min={50} max={200} value={params.threshold} onChange={(v)=>updateParam('threshold', v)} />

            <SliderRow id="minArea" label="Min Area" short={HELP.minArea.short} help={HELP.minArea.help} min={100} max={2000} step={100} value={params.minArea} onChange={(v)=>updateParam('minArea', v)} />

            <SliderRow id="padding" label="Padding" short={HELP.padding.short} help={HELP.padding.help} min={0} max={20} value={params.padding} onChange={(v)=>updateParam('padding', v)} />

            <SliderRow id="morphClose" label="Morph Close" short={HELP.morphClose.short} help={HELP.morphClose.help} min={0} max={5} value={params.morphClose} onChange={(v)=>updateParam('morphClose', v)} />

            <SliderRow id="morphDilate" label="Morph Dilate" short={HELP.morphDilate.short} help={HELP.morphDilate.help} min={0} max={3} value={params.morphDilate} onChange={(v)=>updateParam('morphDilate', v)} />

            <div>
              <div className="text-xs mb-1">Adaptive Block: {params.adaptiveBlock}</div>
              <input
                type="range"
                min="3" max="21" step="2"
                value={params.adaptiveBlock}
                onChange={(e) => setParams({...params, adaptiveBlock: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Adaptive C: {params.adaptiveC}</div>
              <input
                type="range"
                min="0" max="10"
                value={params.adaptiveC}
                onChange={(e) => setParams({...params, adaptiveC: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Approx Eps: {params.approxEps.toFixed(3)}</div>
              <input
                type="range"
                min="0.01" max="0.1" step="0.01"
                value={params.approxEps}
                onChange={(e) => setParams({...params, approxEps: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Min Contour: {params.minContourArea}</div>
              <input
                type="range"
                min="500" max="5000" step="100"
                value={params.minContourArea}
                onChange={(e) => setParams({...params, minContourArea: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Max Contour: {params.maxContourArea}</div>
              <input
                type="range"
                min="10000" max="100000" step="5000"
                value={params.maxContourArea}
                onChange={(e) => setParams({...params, maxContourArea: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Aspect Min: {params.aspectRatioMin.toFixed(1)}</div>
              <input
                type="range"
                min="0.1" max="1.0" step="0.1"
                value={params.aspectRatioMin}
                onChange={(e) => setParams({...params, aspectRatioMin: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            <div>
              <div className="text-xs mb-1">Aspect Max: {params.aspectRatioMax.toFixed(1)}</div>
              <input
                type="range"
                min="1.0" max="5.0" step="0.1"
                value={params.aspectRatioMax}
                onChange={(e) => setParams({...params, aspectRatioMax: Number(e.target.value)})}
                className="w-full h-1"
              />
            </div>

            {/* New Advanced Controls */}
            <div className="border-t border-neutral-700 pt-2 mt-2">
              <div className="text-xs font-semibold mb-2">Advanced Controls</div>
              
              <div>
                <div className="text-xs mb-1">Grid Size: {params.gridSize}</div>
                <input
                  type="range"
                  min="50" max="300" step="10"
                  value={params.gridSize}
                  onChange={(e) => setParams({...params, gridSize: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Sensitivity: {params.sensitivity}</div>
                <input
                  type="range"
                  min="0" max="100"
                  value={params.sensitivity}
                  onChange={(e) => setParams({...params, sensitivity: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Min Density: {params.minDensity}%</div>
                <input
                  type="range"
                  min="0" max="50"
                  value={params.minDensity}
                  onChange={(e) => setParams({...params, minDensity: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Edge Threshold: {params.edgeThreshold}</div>
                <input
                  type="range"
                  min="0" max="200"
                  value={params.edgeThreshold}
                  onChange={(e) => setParams({...params, edgeThreshold: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Brightness: {params.brightness}</div>
                <input
                  type="range"
                  min="-100" max="100"
                  value={params.brightness}
                  onChange={(e) => setParams({...params, brightness: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Contrast: {params.contrastBoost}</div>
                <input
                  type="range"
                  min="-100" max="100"
                  value={params.contrastBoost}
                  onChange={(e) => setParams({...params, contrastBoost: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Gamma: {params.gamma.toFixed(2)}</div>
                <input
                  type="range"
                  min="0.1" max="3.0" step="0.1"
                  value={params.gamma}
                  onChange={(e) => setParams({...params, gamma: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Sharpen: {params.sharpen}</div>
                <input
                  type="range"
                  min="0" max="10"
                  value={params.sharpen}
                  onChange={(e) => setParams({...params, sharpen: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Erosion: {params.erosion}</div>
                <input
                  type="range"
                  min="0" max="5"
                  value={params.erosion}
                  onChange={(e) => setParams({...params, erosion: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Dilation: {params.dilation}</div>
                <input
                  type="range"
                  min="0" max="5"
                  value={params.dilation}
                  onChange={(e) => setParams({...params, dilation: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Opening: {params.opening}</div>
                <input
                  type="range"
                  min="0" max="5"
                  value={params.opening}
                  onChange={(e) => setParams({...params, opening: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>

              <div>
                <div className="text-xs mb-1">Closing: {params.closing}</div>
                <input
                  type="range"
                  min="0" max="5"
                  value={params.closing}
                  onChange={(e) => setParams({...params, closing: Number(e.target.value)})}
                  className="w-full h-1"
                />
              </div>
            </div>

            <button
              onClick={() => setParams({
                blurKernel: 1, cannyLow: 30, cannyHigh: 100, minArea: 500,
                padding: 8, morphClose: 2, morphDilate: 1, threshold: 128,
                adaptiveBlock: 11, adaptiveC: 2, approxEps: 0.02,
                minContourArea: 1000, maxContourArea: 50000,
                aspectRatioMin: 0.3, aspectRatioMax: 3.0,
                gridSize: 150, sensitivity: 50, minDensity: 10, edgeThreshold: 100,
                noiseReduction: 0, contrastBoost: 0, brightness: 0, saturation: 0,
                hueShift: 0, gamma: 1.0, sharpen: 0, erosion: 0, dilation: 0,
                opening: 0, closing: 0
              })}
              className="w-full px-2 py-1 bg-neutral-700 hover:bg-neutral-600 rounded text-xs mt-2"
            >
              Reset All
            </button>
          </div>
        </div>

        {/* Image Display - 45% */}
        <div className="w-[45%] bg-neutral-900 rounded border border-neutral-800 p-4">
          <h3 className="text-lg font-semibold mb-4">Original Image</h3>
          {image ? (
            <div className="relative w-full">
              <img
                ref={imgRef}
                src={image}
                alt="uploaded"
                className="max-w-full max-h-[70vh] object-contain block mx-auto"
                onLoad={(e) => {
                  const el = e.currentTarget
                  setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight })
                  setDisplaySize({ w: el.clientWidth, h: el.clientHeight })
                }}
              />
              {/* Overlay scaled to displayed image size */}
              <div
                className="pointer-events-none absolute left-1/2 -translate-x-1/2"
                style={{ width: displaySize.w, height: displaySize.h, top: 0 }}
              >
                {crops.map((crop) => (
                  crop.polygon ? (
                    <svg
                      key={crop.id}
                      className="absolute left-0 top-0"
                      width={displaySize.w}
                      height={displaySize.h}
                    >
                      <polygon
                        points={crop.polygon.map(([px, py]) => `${Math.round(px * displayScale)},${Math.round(py * displayScale)}`).join(' ')}
                        fill="rgba(255,0,0,0.15)"
                        stroke="#ff0000"
                        strokeWidth={2}
                      />
                    </svg>
                  ) : (
                    <div
                      key={crop.id}
                      className="absolute border-2 border-red-500 opacity-70"
                      style={{
                        left: Math.round(crop.bounds.x * displayScale),
                        top: Math.round(crop.bounds.y * displayScale),
                        width: Math.round(crop.bounds.w * displayScale),
                        height: Math.round(crop.bounds.h * displayScale)
                      }}
                    />
                  )
                ))}
              </div>
            </div>
          ) : (
            <div className="h-64 flex items-center justify-center text-neutral-500">
              Upload an image to start
            </div>
          )}
        </div>

        {/* Crops Display - 45% */}
        <div className="w-[45%] bg-neutral-900 rounded border border-neutral-800 p-4">
          <h3 className="text-lg font-semibold mb-4">Cropped Items</h3>
          <div className="grid grid-cols-3 gap-2 max-h-full overflow-auto">
            {crops.map((crop) => (
              <div key={crop.id} className="bg-neutral-800 rounded p-2">
                <img
                  src={URL.createObjectURL(crop.blob)}
                  alt={crop.name}
                  className="w-full h-24 object-cover rounded mb-2"
                />
                <div className="flex items-center justify-between gap-2">
                  {renamingId === crop.id ? (
                    <input
                      autoFocus
                      value={renameInput}
                      onChange={(e) => setRenameInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          const next = renameInput.trim()
                          const ok = /^[-0-9]+$/.test(next)
                          if (ok) {
                            setCrops(crops.map(c => c.id === crop.id ? { ...c, name: next, needsReview: false } : c))
                            setRenamingId(null)
                          }
                        } else if (e.key === 'Escape') {
                          setRenamingId(null)
                        }
                      }}
                      className="text-xs bg-neutral-700 border border-neutral-600 rounded px-1 py-0.5 w-24"
                    />
                  ) : (
                    <button
                      onDoubleClick={() => { setRenamingId(crop.id); setRenameInput(crop.name) }}
                      title={crop.needsReview ? 'Needs review' : 'Double-click to rename'}
                      className={`text-xs text-left truncate ${crop.needsReview ? 'text-amber-300' : ''}`}
                      style={{ maxWidth: '7rem' }}
                    >
                      {crop.name}
                      {crop.needsReview && <span className="ml-1">⚠︎</span>}
                    </button>
                  )}
                  <button
                    onClick={() => downloadCrop(crop)}
                    className="px-2 py-1 bg-blue-600 hover:bg-blue-500 rounded text-xs"
                  >
                    ↓
                  </button>
                </div>
              </div>
            ))}
          </div>
          {crops.length === 0 && (
            <div className="h-64 flex items-center justify-center text-neutral-500">
              No crops yet
            </div>
          )}
        </div>
      </div>

      {/* Hidden canvas for processing */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default App