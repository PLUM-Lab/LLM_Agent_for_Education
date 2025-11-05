# ChatGPT Integration Guide

This document explains how to use the ChatGPT feature in the Feedback section.

## Features

- 💬 Ask ChatGPT anything about the question or feedback
- 🔄 Real-time chat interface
- 📝 Conversation history maintained
- 🎯 Context-aware responses (includes question and feedback context)

## How to Use

### Step 1: Submit an Answer
1. Select an answer and click "Submit"
2. The Feedback section will appear with explanations
3. The ChatGPT chat box will appear below the feedback

### Step 2: Ask Questions
1. Type your question in the chat input box
2. Click "Send" or press Enter
3. ChatGPT will respond with helpful information

## Configuration Options

### Option 1: Backend API (Recommended for Production)

**Why use backend?**
- ✅ Keeps API key secure (never exposed to browser)
- ✅ Better error handling
- ✅ Can add authentication
- ✅ Can cache responses
- ✅ No CORS issues

**Setup Steps:**

1. **Create Backend Endpoint**
   - See `backend-chat-example.js` for a Node.js/Express example
   - Endpoint should accept POST requests at `/api/chat`
   - Expected request format:
     ```json
     {
       "message": "What is appendicitis?",
       "context": {
         "question": "...",
         "feedback": {...}
       },
       "history": [...]
     }
     ```
   - Expected response format:
     ```json
     {
       "success": true,
       "response": "ChatGPT response here",
       "message": "ChatGPT response here"
     }
     ```

2. **Update Frontend**
   - The frontend is already configured to call `/api/chat`
   - Make sure your backend server is running
   - Update the fetch URL in `medical-quiz.html` if your endpoint is different

3. **Example Backend (Node.js/Express)**
   ```bash
   # Install dependencies
   npm install express axios
   
   # Set API key
   export OPENAI_API_KEY=your-key-here
   
   # Run server
   node backend-chat-example.js
   ```

### Option 2: Direct OpenAI API (Development Only)

**⚠️ Warning:** This exposes your API key in the browser. Only use for development!

**Setup Steps:**

1. Check the "Use OpenAI API directly" checkbox in the Feedback section
2. When you send a message, you'll be prompted for your API key
3. Enter your OpenAI API key (it's only used for that session)

**Security Note:** 
- Never commit API keys to version control
- Never use this method in production
- Always use a backend proxy for production apps

## API Requirements

### OpenAI API Key
- Get your API key from: https://platform.openai.com/api-keys
- Keep it secure and never expose it in client-side code

### Backend Endpoint
- Method: POST
- Path: `/api/chat` (or configure in frontend)
- Headers: `Content-Type: application/json`
- Request Body:
  ```json
  {
    "message": "User's question",
    "context": {
      "question": "Current question text",
      "feedback": { ... }
    },
    "history": [
      { "role": "user", "content": "..." },
      { "role": "assistant", "content": "..." }
    ]
  }
  ```
- Response:
  ```json
  {
    "success": true,
    "response": "ChatGPT's answer",
    "message": "ChatGPT's answer"
  }
  ```

## Customization

### Change Model
In `medical-quiz.html`, find `callOpenAIDirect()` function and change:
```javascript
model: 'gpt-3.5-turbo',  // Change to 'gpt-4' or other models
```

### Adjust Response Length
Change `max_tokens`:
```javascript
max_tokens: 500,  // Increase for longer responses
```

### Customize System Prompt
Modify the system message in both `callOpenAIDirect()` and your backend:
```javascript
content: 'You are a helpful medical education assistant...'
```

## Troubleshooting

### "Backend API error"
- Check if your backend server is running
- Verify the endpoint URL is correct
- Check browser console for detailed error messages

### "API key required"
- Make sure you've entered a valid OpenAI API key
- Check that the API key has sufficient credits

### CORS Errors
- Configure CORS on your backend server
- Or use a backend proxy (recommended)

### No Response
- Check browser console (F12) for errors
- Verify API key is valid
- Check network tab for API request status

## Example Questions to Ask

- "What is appendicitis?"
- "Why is ultrasound preferred over CT scan here?"
- "Can you explain the differential diagnosis?"
- "What are the complications if not treated?"
- "What are the signs and symptoms?"

## Security Best Practices

1. **Never expose API keys in frontend code**
2. **Use environment variables for API keys**
3. **Implement rate limiting on backend**
4. **Add authentication if needed**
5. **Validate and sanitize user input**
6. **Monitor API usage and costs**

