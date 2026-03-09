# Medical Quiz Application - Deployment Package

This folder contains all necessary resources for deploying the Medical Quiz application.

## Files Included

### Core Application
- **medical-quiz.html** - Main application file (single-page web application)

### Configuration
- **../api-key.js** - OpenAI API key configuration (optional, can use localStorage instead)
  - If this file doesn't exist, the application will prompt for API key on first use
  - You can create this file with: `const OPENAI_API_KEY = 'your-api-key-here';`

### Question Bank Data Files
- **questions.json** - Clinical questions generated from medical guidelines
- **qbank_combined.json** - Combined question bank (Surgery + Amboss)
- **generated_surgery_domain_questions.json** - Surgery domain questions (579 questions)

## Deployment Instructions

### Option 1: Simple HTTP Server (Recommended for Testing)

1. Navigate to this folder:
   ```bash
   cd LLM_Agent_for_Education
   ```

2. Start a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   
   # Python 2
   python -m SimpleHTTPServer 8000
   
   # Node.js (if you have http-server installed)
   npx http-server -p 8000
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:8000/medical-quiz.html
   ```

### Option 2: Web Server Deployment

1. Copy all files in this folder to your web server's document root (e.g., `/var/www/html/` or `C:\inetpub\wwwroot\`)

2. Ensure your web server is configured to serve HTML files

3. Access the application via your web server URL:
   ```
   http://your-server-domain/medical-quiz.html
   ```

### Option 3: Static Hosting (GitHub Pages, Netlify, Vercel, etc.)

1. Upload all files in this folder (plus api-key.js in the parent folder) to your static hosting service

2. Set `medical-quiz.html` as your entry point/index file

3. The application will be accessible via your hosting service URL

## Backend Services (Optional)

The application can work standalone, but for full functionality, you may want to run:

### RAG Server (Optional)
- Provides semantic search and enhanced context retrieval
- See main project's `rag_server.py` for backend server
- Default port: 5000
- The application will gracefully degrade if the RAG server is unavailable

## Features

- âœ?Standalone deployment (all resources in one folder)
- âœ?No external dependencies (except optional API key)
- âœ?Works offline (after initial load)
- âœ?Responsive design (mobile and desktop)
- âœ?Multiple question banks
- âœ?Student progress tracking (localStorage)
- âœ?AI tutor integration (OpenAI API)

## Notes

- All file paths are relative, so the folder structure must be maintained
- The application uses localStorage for student profiles (browser-specific)
- API key can be provided via `api-key.js` or entered manually in the UI
- Question bank files are loaded dynamically based on user selection

## Troubleshooting

### Questions not loading?
- Check browser console for errors
- Ensure JSON files are accessible (check file permissions)
- Verify web server is serving JSON files with correct MIME type (`application/json`)

### API key issues?
- Create `api-key.js` file with: `const OPENAI_API_KEY = 'your-key';`
- Or enter API key manually when prompted in the application

### CORS errors?
- Ensure you're accessing via HTTP server, not `file://` protocol
- Check that your web server allows CORS if accessing from different domain
