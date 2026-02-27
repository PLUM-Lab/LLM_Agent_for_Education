# Deployment Checklist

## ✅ Files Included in Deployment Package

### Core Application
- [x] `medical-quiz.html` - Main application (365 KB)

### Configuration Files
- [x] `api-key.js` - API key configuration (optional)
- [x] `api-key.example.js` - Example API key file template

### Question Bank Data Files
- [x] `questions.json` - Clinical questions (591 KB)
- [x] `qbank_combined.json` - Combined question bank (882 KB)
- [x] `generated_surgery_domain_questions.json` - Surgery questions (685 KB)

### Documentation
- [x] `README.md` - Deployment instructions
- [x] `.gitignore` - Git ignore rules

## 📋 Pre-Deployment Checklist

- [ ] Verify all files are present in the `deploy` folder
- [ ] Check that `api-key.js` exists (or create from `api-key.example.js`)
- [ ] Test locally using `python -m http.server 8000`
- [ ] Verify questions load correctly
- [ ] Test login/registration functionality
- [ ] Verify API key entry works

## 🚀 Deployment Steps

1. **Copy all files** from `deploy` folder to your web server
2. **Set permissions** (if needed):
   - HTML files: readable by web server
   - JSON files: readable by web server
   - JavaScript files: readable by web server
3. **Configure web server** to serve JSON files with correct MIME type
4. **Test deployment**:
   - Access `medical-quiz.html` via web browser
   - Verify questions load
   - Test login/registration
   - Test question answering
   - Test tutor chat functionality

## 🔧 Optional Backend Services

- [ ] RAG Server (`rag_server.py`) - For enhanced semantic search
- [ ] Port 5000 - RAG server port (if using RAG features)

## 📝 Post-Deployment Verification

- [ ] Application loads without errors
- [ ] Questions display correctly
- [ ] Student can answer questions
- [ ] Feedback displays correctly
- [ ] Tutor chat works (requires API key)
- [ ] Progress tracking works (localStorage)
- [ ] Multiple question banks load correctly

## 🐛 Troubleshooting

If issues occur:
1. Check browser console for errors
2. Verify all files are accessible
3. Check web server logs
4. Verify JSON files have correct MIME type
5. Check CORS settings if accessing from different domain
