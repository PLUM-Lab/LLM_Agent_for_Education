# Student Quiz System UI

This is a complete student quiz system UI interface, containing three different state interfaces.

## Interface States

### 1.1 Before Student Submits Answer (`data-before-submit.json`)
- Display question information (question number, type, title, content)
- Provide answer input area (supports single choice, multiple choice, text input, short answer, etc.)
- Submit answer and reset buttons

### 1.2 After Student Submits Correct Answer (`data-after-correct.json`)
- Display success icon and "Correct Answer!" message
- Display question information
- Display user answer and correct answer (both highlighted in green)
- Display detailed explanation
- Next question and view details buttons

### 1.3 After Student Submits Wrong Answer (`data-after-wrong.json`)
- Display error icon and "Incorrect Answer" message
- Display question information
- Display user answer (highlighted in red) and correct answer (highlighted in green)
- Display detailed explanation and hints
- Try again and next question buttons

## File Structure

```
.
├── index.html              # Main HTML file containing three state interfaces
├── styles.css              # CSS stylesheet file
├── app.js                  # JavaScript logic file
├── data-before-submit.json # JSON data sample before submission
├── data-after-correct.json # JSON data sample after correct answer
├── data-after-wrong.json   # JSON data sample after wrong answer
└── README.md               # This documentation file
```

## Usage

### 1. Direct Open
Simply open the `index.html` file directly in your browser to view the interface.

### 2. Using Local Server (Recommended)
Since JSON file loading is involved, it's recommended to run using a local server:

```bash
# Using Python (Python 3)
python -m http.server 8000

# Using Python 2
python -m SimpleHTTPServer 8000

# Using Node.js http-server
npx http-server -p 8000

# Using PHP
php -S localhost:8000
```

Then visit `http://localhost:8000` in your browser.

## JSON Data Format

### Before Submit Data Format (`data-before-submit.json`)

```json
{
  "state": "before",
  "question": {
    "number": 1,
    "type": "Multiple Choice",
    "title": "Question Title",
    "content": "Question Content"
  },
  "answerOptions": [
    {
      "value": "A",
      "label": "A. Option Content"
    }
  ],
  "metadata": {
    "difficulty": "Easy",
    "points": 5,
    "timeLimit": 300
  }
}
```

### After Correct Answer Data Format (`data-after-correct.json`)

```json
{
  "state": "correct",
  "question": {
    "number": 1,
    "type": "Multiple Choice",
    "title": "Question Title",
    "content": "Question Content"
  },
  "userAnswer": "D",
  "correctAnswer": "D",
  "isCorrect": true,
  "explanation": "Explanation content",
  "score": {
    "earned": 5,
    "total": 5
  },
  "feedback": "Feedback message"
}
```

### After Wrong Answer Data Format (`data-after-wrong.json`)

```json
{
  "state": "wrong",
  "question": {
    "number": 1,
    "type": "Multiple Choice",
    "title": "Question Title",
    "content": "Question Content"
  },
  "userAnswer": "A",
  "correctAnswer": "D",
  "isCorrect": false,
  "explanation": "Explanation content",
  "score": {
    "earned": 0,
    "total": 5
  },
  "feedback": "Feedback message",
  "hints": ["Hint 1", "Hint 2"]
}
```

## Backend Integration

### Method 1: Via API Call

Modify the `handleSubmit` function in `app.js` to change the submission logic to call the backend API:

```javascript
async function handleSubmit() {
    const userAnswer = getUserAnswer();
    
    if (!userAnswer) {
        alert('Please select or enter an answer first');
        return;
    }

    try {
        const response = await fetch('/api/submit-answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                questionId: currentQuestionData.question.id,
                answer: userAnswer
            })
        });
        
        const result = await response.json();
        
        if (result.isCorrect) {
            showCorrectState(result, userAnswer);
        } else {
            showWrongState(result, userAnswer);
        }
    } catch (error) {
        console.error('Submission failed:', error);
        alert('Submission failed. Please try again later.');
    }
}
```

### Method 2: Direct Data Setting

The backend can directly set question data via the `window.setQuestionData` function:

```javascript
// After backend returns data
window.setQuestionData({
    state: 'before', // or 'correct', 'wrong'
    question: { ... },
    // ... other data
});
```

## Supported Question Types

- **Multiple Choice** (`type: "Multiple Choice"` or `"single"`): Radio buttons
- **Multiple Select** (`type: "Multiple Select"` or `"multiple"`): Checkboxes
- **Fill in the Blank** (`type: "Fill in the Blank"` or `"text"`): Text input
- **Short Answer** (`type: "Short Answer"` or `"essay"`): Textarea

## Features

- ✨ Modern gradient background design
- 📱 Responsive layout, mobile-friendly
- 🎨 Beautiful UI interface with animation effects
- 🔄 Smooth state transitions
- 📝 Support for multiple question types
- 🎯 Clear answer comparison display
- 💡 Detailed explanations and hints

## Browser Compatibility

Supports all modern browsers:
- Chrome (Recommended)
- Firefox
- Safari
- Edge

## Customization

You can customize styles by modifying the `styles.css` file:
- Change color theme (search for `#667eea` and `#764ba2`)
- Adjust spacing and sizing
- Modify animation effects

## Notes

1. If you open `index.html` directly in the browser (using `file://` protocol), you may encounter CORS restrictions preventing JSON file loading. It's recommended to use a local server.
2. The `content` and `explanation` fields in JSON files support HTML format, including tags like `<br>`, `<p>`, `<strong>`, etc.
3. Multiple choice answers are formatted as arrays, e.g., `["A", "B"]`.
