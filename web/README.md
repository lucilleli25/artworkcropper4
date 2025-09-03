# Tattoo Sheet Cropper

A client-side React + Vite + TypeScript app for detecting and cropping individual tattoos from scanned sheets. Uses OpenCV.js in Web Workers for computer vision processing and optional Tesseract.js for OCR-based auto-naming.

## Features

- **Automatic Detection**: Detects individual tattoos using OpenCV.js with multiple algorithms (Otsu, Adaptive, Canny, Auto)
- **Loose Boundaries**: Creates padded polygon boundaries to ensure full tattoo capture
- **Separation**: Advanced watershed-based separation to split touching tattoos
- **OCR Auto-naming**: Optional digits-only OCR to extract numeric labels for automatic naming
- **Interactive Editing**: Rename, delete, and toggle items before export
- **Bulk Export**: Export selected items as transparent PNGs in a ZIP file
- **Real-time Preview**: Live polygon overlay and cropped image preview
- **Parameter Tuning**: Comprehensive controls for fine-tuning detection parameters

## Tech Stack

- **UI**: React + Vite + TypeScript
- **Styling**: Tailwind CSS
- **State**: Zustand
- **Computer Vision**: OpenCV.js (Web Worker)
- **OCR**: Tesseract.js (Web Worker, optional)
- **Export**: JSZip
- **File Handling**: createImageBitmap, OffscreenCanvas

## Installation

```bash
cd web
npm install
```

## Development

```bash
npm run dev
```

Opens the app at `http://localhost:5173`

## Build

```bash
npm run build
```

## Preview

```bash
npm run preview
```

## Usage

1. **Upload a Sheet**: Click "Upload Sheet" to load a PNG/JPG image
2. **Detect Tattoos**: Click "Detect Tattoos" to run the detection pipeline
3. **Adjust Parameters**: Use the left panel controls to fine-tune detection
4. **Review Results**: Check the right panel for detected items and cropped images
5. **Edit Items**: Rename, delete, or toggle items as needed
6. **Export**: Click "Export ZIP" to download selected items

## Detection Pipeline

1. **Resize**: Downsize image for processing while maintaining aspect ratio
2. **Color Normalization**: Optional CLAHE and grayworld correction
3. **Mask Creation**: Choose best method (Otsu/Adaptive/Canny/Auto)
4. **Morphology**: Close gaps and optionally dilate
5. **Separation**: Watershed-based separation of touching components
6. **Contour Detection**: Find external contours and filter by area
7. **Polygonization**: Approximate contours to polygons with configurable epsilon
8. **Padding**: Dilate boundaries for loose capture
9. **NMS**: Remove overlapping detections
10. **Crop**: Create transparent PNG crops using OffscreenCanvas

## Parameters

### Detection Method
- **Auto**: Automatically chooses the best method based on image content
- **Otsu**: Global thresholding, good for high contrast images
- **Adaptive**: Local thresholding, handles uneven lighting
- **Canny**: Edge detection, good for line art

### Preprocessing
- **Blur Kernel**: Gaussian blur size (odd numbers only)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization

### Morphology
- **Kernel Size**: Structuring element size for morphological operations
- **Close Iterations**: Number of closing operations to fill gaps
- **Dilate Iterations**: Number of dilation operations to thicken

### Separation
- **Enable**: Toggle watershed-based separation
- **Min Peak Distance**: Minimum distance between peaks for separation

### Filtering
- **Min Area %**: Minimum area as percentage of image for valid detections
- **Poly Epsilon %**: Polygon approximation tolerance
- **Padding Px**: Extra padding around detected boundaries

### Performance
- **Fast Mode**: Limit processing resolution for speed
- **Max Side**: Maximum dimension for processing

### OCR
- **Enable Digits**: Toggle OCR for automatic naming
- **Confidence**: Minimum confidence threshold for OCR results
- **Ring Inner Px**: Inner radius of text detection ring
- **Ring Width Px**: Width of text detection ring

## File Structure

```
web/
├── src/
│   ├── components/
│   │   ├── SheetCanvas.tsx      # Image display with SVG overlay
│   │   ├── ControlsPanel.tsx    # Parameter controls
│   │   ├── ResultsGrid.tsx      # Detected items grid
│   │   └── TopBar.tsx           # Main navigation
│   ├── hooks/
│   │   └── useWorkers.ts        # Worker orchestration
│   ├── store/
│   │   └── store.ts             # Zustand state management
│   ├── utils/
│   │   ├── image.ts             # Image utilities
│   │   └── zip.ts               # Export utilities
│   └── App.tsx                  # Main app component
├── public/
│   ├── visionWorker.js          # OpenCV.js detection worker
│   └── ocrWorker.js             # Tesseract.js OCR worker
└── README.md
```

## Performance

- **Fast Mode**: 0.5-1.5s on 3000×4000px sheets
- **With OCR**: +0.4-0.9s depending on item count
- **Memory**: Efficient cleanup after processing
- **Responsive**: All processing in Web Workers

## Browser Support

- Modern browsers with Web Workers and OffscreenCanvas support
- Chrome 69+, Firefox 105+, Safari 16.4+

## License

MIT
