// Example backend API endpoint for ChatGPT integration
// This is a Node.js/Express example - adapt to your backend framework

const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// Your OpenAI API key (store in environment variable for security)
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || 'your-api-key-here';

// CORS middleware (if needed)
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Headers', 'Content-Type');
    next();
});

// ChatGPT API endpoint
app.post('/api/chat', async (req, res) => {
    try {
        const { message, context, history } = req.body;

        // Build system prompt with context
        const systemPrompt = `You are a helpful medical education assistant. 
Answer questions about medical cases clearly and accurately.
${context?.question ? `Current question: ${context.question}` : ''}
${context?.feedback?.correctness ? `Feedback: ${context.feedback.correctness}` : ''}`;

        // Prepare messages for OpenAI
        const messages = [
            {
                role: 'system',
                content: systemPrompt
            },
            // Add conversation history if provided
            ...(history || []),
            // Add current user message
            {
                role: 'user',
                content: message
            }
        ];

        // Call OpenAI API
        const response = await axios.post('https://api.openai.com/v1/chat/completions', {
            model: 'gpt-3.5-turbo',
            messages: messages,
            max_tokens: 500,
            temperature: 0.7
        }, {
            headers: {
                'Authorization': `Bearer ${OPENAI_API_KEY}`,
                'Content-Type': 'application/json'
            }
        });

        const assistantMessage = response.data.choices[0].message.content;

        // Return response
        res.json({
            success: true,
            response: assistantMessage,
            message: assistantMessage // Alternative field name
        });

    } catch (error) {
        console.error('ChatGPT API Error:', error.response?.data || error.message);
        res.status(500).json({
            success: false,
            error: 'Failed to get response from ChatGPT',
            message: 'Please check your API configuration'
        });
    }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`ChatGPT endpoint: http://localhost:${PORT}/api/chat`);
});

// Example usage:
// 1. Install dependencies: npm install express axios
// 2. Set environment variable: export OPENAI_API_KEY=your-key-here
// 3. Run: node backend-chat-example.js
// 4. Update frontend to use: http://localhost:3000/api/chat

