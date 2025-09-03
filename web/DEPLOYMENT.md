# Deployment Guide

This guide covers multiple deployment options for the Tattoo Sheet Cropper app.

## Prerequisites

1. **Install dependencies**:
   ```bash
   cd web
   npm install
   ```

2. **Build the app**:
   ```bash
   npm run build
   ```

This creates a `dist/` folder with all the static files ready for deployment.

## Deployment Options

### 1. Vercel (Recommended)

**Pros**: Easy setup, automatic HTTPS, global CDN, free tier
**Best for**: Quick deployment with minimal configuration

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Deploy**:
   ```bash
   vercel --prod
   ```

3. **Or connect to GitHub**:
   - Push your code to GitHub
   - Go to [vercel.com](https://vercel.com)
   - Import your repository
   - Deploy automatically

**Configuration**: No additional config needed. Vercel automatically detects Vite.

### 2. Netlify

**Pros**: Easy drag-and-drop, form handling, serverless functions
**Best for**: Simple static hosting with additional features

1. **Install Netlify CLI**:
   ```bash
   npm i -g netlify-cli
   ```

2. **Deploy**:
   ```bash
   netlify deploy --prod --dir=dist
   ```

3. **Or drag-and-drop**:
   - Go to [netlify.com](https://netlify.com)
   - Drag the `dist/` folder to the deploy area

**Configuration**: Create `netlify.toml`:
```toml
[build]
  publish = "dist"
  command = "npm run build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

### 3. GitHub Pages

**Pros**: Free with GitHub, automatic updates
**Best for**: Open source projects

1. **Install gh-pages**:
   ```bash
   npm install --save-dev gh-pages
   ```

2. **Add to package.json**:
   ```json
   {
     "scripts": {
       "predeploy": "npm run build",
       "deploy": "gh-pages -d dist"
     }
   }
   ```

3. **Deploy**:
   ```bash
   npm run deploy
   ```

4. **Enable GitHub Pages**:
   - Go to repository Settings → Pages
   - Select "Deploy from a branch"
   - Choose `gh-pages` branch

### 4. Firebase Hosting

**Pros**: Google infrastructure, easy setup, good performance
**Best for**: Google ecosystem integration

1. **Install Firebase CLI**:
   ```bash
   npm install -g firebase-tools
   ```

2. **Login and init**:
   ```bash
   firebase login
   firebase init hosting
   ```

3. **Configure**:
   - Public directory: `dist`
   - Single-page app: Yes
   - Overwrite index.html: No

4. **Deploy**:
   ```bash
   firebase deploy
   ```

### 5. AWS S3 + CloudFront

**Pros**: Highly scalable, cost-effective for large traffic
**Best for**: Production applications with high traffic

1. **Install AWS CLI**:
   ```bash
   aws configure
   ```

2. **Create S3 bucket**:
   ```bash
   aws s3 mb s3://your-bucket-name
   ```

3. **Upload files**:
   ```bash
   aws s3 sync dist/ s3://your-bucket-name --delete
   ```

4. **Enable static website hosting**:
   ```bash
   aws s3 website s3://your-bucket-name --index-document index.html --error-document index.html
   ```

5. **Set up CloudFront** (optional, for CDN):
   - Create CloudFront distribution
   - Point to S3 bucket
   - Configure custom error pages

### 6. Docker + Any Host

**Pros**: Consistent deployment, easy scaling
**Best for**: Containerized environments

1. **Create Dockerfile**:
   ```dockerfile
   FROM nginx:alpine
   COPY dist/ /usr/share/nginx/html/
   COPY nginx.conf /etc/nginx/nginx.conf
   EXPOSE 80
   CMD ["nginx", "-g", "daemon off;"]
   ```

2. **Create nginx.conf**:
   ```nginx
   events {
       worker_connections 1024;
   }
   
   http {
       include /etc/nginx/mime.types;
       default_type application/octet-stream;
       
       server {
           listen 80;
           server_name localhost;
           root /usr/share/nginx/html;
           index index.html;
           
           location / {
               try_files $uri $uri/ /index.html;
           }
       }
   }
   ```

3. **Build and run**:
   ```bash
   docker build -t tattoo-cropper .
   docker run -p 80:80 tattoo-cropper
   ```

## Environment-Specific Considerations

### Production Build Optimization

1. **Enable compression** (gzip/brotli)
2. **Set proper cache headers** for static assets
3. **Use HTTPS** for security
4. **Configure CSP headers** if needed

### Worker Files

The app uses Web Workers (`visionWorker.js` and `ocrWorker.js`) that load external libraries:
- OpenCV.js from CDN
- Tesseract.js from CDN

Ensure your deployment platform allows:
- Loading external scripts
- Web Worker execution
- OffscreenCanvas support

### Browser Compatibility

The app requires:
- Modern browsers with Web Workers
- OffscreenCanvas support
- ES2020+ features

Test on:
- Chrome 69+
- Firefox 105+
- Safari 16.4+

## Quick Start (Recommended)

For the fastest deployment:

```bash
# 1. Install dependencies
cd web
npm install

# 2. Build the app
npm run build

# 3. Deploy to Vercel (easiest)
npm i -g vercel
vercel --prod
```

Your app will be live at a Vercel URL within minutes!

## Troubleshooting

### Build Issues
- Ensure all dependencies are installed: `npm install`
- Check TypeScript errors: `npm run build`
- Verify all imports are correct

### Runtime Issues
- Check browser console for errors
- Verify Web Workers are loading correctly
- Ensure external CDN resources are accessible

### Performance Issues
- Enable gzip compression on your server
- Use a CDN for static assets
- Consider lazy loading for large images

## Security Considerations

1. **Content Security Policy**: Configure CSP headers to allow external scripts
2. **HTTPS**: Always use HTTPS in production
3. **Worker Security**: Web Workers run in isolated contexts
4. **File Upload**: Client-side only, no server processing

## Monitoring

Consider adding:
- Error tracking (Sentry, LogRocket)
- Analytics (Google Analytics, Plausible)
- Performance monitoring (Web Vitals)
