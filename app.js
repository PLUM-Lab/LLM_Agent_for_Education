// Current state management
let currentState = 'before'; // 'before', 'correct', 'wrong'
let currentQuestionData = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    // Load default before-submit state data
    loadQuestionData('data-before-submit.json');
});

// Initialize event listeners
function initializeEventListeners() {
    // Submit button
    document.getElementById('submit-btn').addEventListener('click', handleSubmit);
    
    // Reset button
    document.getElementById('reset-btn').addEventListener('click', handleReset);
    
    // Next button (correct state)
    document.getElementById('next-btn').addEventListener('click', handleNext);
    
    // Review button
    document.getElementById('review-btn').addEventListener('click', handleReview);
    
    // Retry button (wrong state)
    document.getElementById('retry-btn').addEventListener('click', handleRetry);
    
    // Next button (wrong state)
    document.getElementById('next-question-btn').addEventListener('click', handleNext);
}

// Load question data
async function loadQuestionData(filename) {
    try {
        const response = await fetch(filename);
        const data = await response.json();
        currentQuestionData = data;
        renderQuestion(data);
    } catch (error) {
        console.error('Failed to load data:', error);
        alert('Failed to load question data. Please check if the JSON file exists.');
    }
}

// Render question (before submit state)
function renderQuestion(data) {
    if (!data || !data.question) return;

    const question = data.question;
    
    // Update question information
    document.getElementById('question-num-before').textContent = question.number || '1';
    document.getElementById('question-type-before').textContent = question.type || 'Multiple Choice';
    document.getElementById('question-title-before').textContent = question.title || 'Question';
    document.getElementById('question-body-before').innerHTML = formatContent(question.content);

    // Render answer input area
    renderAnswerInput(question, data.answerOptions);
}

// Render answer input area
function renderAnswerInput(question, answerOptions) {
    const container = document.getElementById('answer-input-container');
    container.innerHTML = '';

    if (!answerOptions) return;

    const questionType = question.type || 'single';

    if (questionType === 'single' || questionType === 'Multiple Choice' || questionType === '选择题') {
        // Single choice question
        answerOptions.forEach((option, index) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'radio-option';
            optionDiv.innerHTML = `
                <input type="radio" name="answer" id="option-${index}" value="${option.value || option.label || option}">
                <label for="option-${index}">${option.label || option}</label>
            `;
            optionDiv.addEventListener('click', function(e) {
                if (e.target.type !== 'radio') {
                    this.querySelector('input[type="radio"]').checked = true;
                    updateRadioSelection();
                }
            });
            container.appendChild(optionDiv);
        });
    } else if (questionType === 'multiple' || questionType === 'Multiple Select' || questionType === '多选题') {
        // Multiple choice question
        answerOptions.forEach((option, index) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'checkbox-option';
            optionDiv.innerHTML = `
                <input type="checkbox" name="answer" id="option-${index}" value="${option.value || option.label || option}">
                <label for="option-${index}">${option.label || option}</label>
            `;
            optionDiv.addEventListener('click', function(e) {
                if (e.target.type !== 'checkbox') {
                    const checkbox = this.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                    updateCheckboxSelection();
                }
            });
            container.appendChild(optionDiv);
        });
    } else if (questionType === 'text' || questionType === 'Fill in the Blank' || questionType === '填空题') {
        // Text input
        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'text-input';
        input.placeholder = 'Enter your answer';
        input.name = 'answer';
        container.appendChild(input);
    } else if (questionType === 'essay' || questionType === 'Short Answer' || questionType === '简答题') {
        // Textarea input
        const textarea = document.createElement('textarea');
        textarea.className = 'textarea-input';
        textarea.placeholder = 'Enter your answer';
        textarea.name = 'answer';
        container.appendChild(textarea);
    }
}

// Update radio button selection style
function updateRadioSelection() {
    document.querySelectorAll('.radio-option').forEach(option => {
        option.classList.remove('checked');
        if (option.querySelector('input[type="radio"]').checked) {
            option.classList.add('checked');
        }
    });
}

// Update checkbox selection style
function updateCheckboxSelection() {
    document.querySelectorAll('.checkbox-option').forEach(option => {
        option.classList.remove('checked');
        if (option.querySelector('input[type="checkbox"]').checked) {
            option.classList.add('checked');
        }
    });
}

// Get user answer
function getUserAnswer() {
    const questionType = currentQuestionData.question.type || 'single';
    
    if (questionType === 'single' || questionType === 'Multiple Choice' || questionType === '选择题') {
        const selected = document.querySelector('input[name="answer"]:checked');
        return selected ? selected.value : null;
    } else if (questionType === 'multiple' || questionType === 'Multiple Select' || questionType === '多选题') {
        const selected = document.querySelectorAll('input[name="answer"]:checked');
        return Array.from(selected).map(input => input.value);
    } else {
        const input = document.querySelector('input[name="answer"], textarea[name="answer"]');
        return input ? input.value.trim() : '';
    }
}

// Handle submit
async function handleSubmit() {
    const userAnswer = getUserAnswer();
    
    if (!userAnswer || (Array.isArray(userAnswer) && userAnswer.length === 0)) {
        alert('Please select or enter an answer first');
        return;
    }

    // Here you should call the backend API, but for demo purposes, we directly load the corresponding JSON file
    // In actual application, you should load different state data based on the backend response
    
    // Simulate: load different data based on answer correctness
    // Actual implementation: const response = await fetch('/api/submit-answer', { method: 'POST', body: JSON.stringify({ answer: userAnswer }) });
    // const result = await response.json();
    
    // Here we assume the backend will return an identifier, then load the corresponding JSON file
    // For demo purposes, we use a simple judgment logic
    try {
        // Try to load correct answer data
        const correctData = await fetch('data-after-correct.json').catch(() => null);
        if (correctData && correctData.ok) {
            const data = await correctData.json();
            showCorrectState(data, userAnswer);
            return;
        }
    } catch (e) {
        // If correct data doesn't exist, load wrong data
        try {
            const wrongData = await fetch('data-after-wrong.json');
            const data = await wrongData.json();
            showWrongState(data, userAnswer);
        } catch (err) {
            console.error('Failed to load result data:', err);
            alert('Submission failed. Please check the data file.');
        }
    }
}

// Show correct state
function showCorrectState(data, userAnswer) {
    currentState = 'correct';
    
    // Hide all states
    document.querySelectorAll('.state-container').forEach(el => el.classList.add('hidden'));
    
    // Show correct state
    const correctContainer = document.getElementById('state-correct');
    correctContainer.classList.remove('hidden');
    
    // Populate data
    const question = data.question || currentQuestionData.question;
    document.getElementById('question-num-correct').textContent = question.number || '1';
    document.getElementById('question-type-correct').textContent = question.type || 'Multiple Choice';
    document.getElementById('question-title-correct').textContent = question.title || 'Question';
    document.getElementById('question-body-correct').innerHTML = formatContent(question.content);
    
    // Display answers
    document.getElementById('user-answer-correct').textContent = formatAnswer(userAnswer);
    document.getElementById('correct-answer-display').textContent = formatAnswer(data.correctAnswer || data.answer);
    
    // Display explanation
    if (data.explanation) {
        document.getElementById('explanation-correct').innerHTML = formatContent(data.explanation);
    }
}

// Show wrong state
function showWrongState(data, userAnswer) {
    currentState = 'wrong';
    
    // Hide all states
    document.querySelectorAll('.state-container').forEach(el => el.classList.add('hidden'));
    
    // Show wrong state
    const wrongContainer = document.getElementById('state-wrong');
    wrongContainer.classList.remove('hidden');
    
    // Populate data
    const question = data.question || currentQuestionData.question;
    document.getElementById('question-num-wrong').textContent = question.number || '1';
    document.getElementById('question-type-wrong').textContent = question.type || 'Multiple Choice';
    document.getElementById('question-title-wrong').textContent = question.title || 'Question';
    document.getElementById('question-body-wrong').innerHTML = formatContent(question.content);
    
    // Display answers
    document.getElementById('user-answer-wrong').textContent = formatAnswer(userAnswer);
    document.getElementById('correct-answer-wrong').textContent = formatAnswer(data.correctAnswer || data.answer);
    
    // Display explanation
    if (data.explanation) {
        document.getElementById('explanation-wrong').innerHTML = formatContent(data.explanation);
    }
}

// Format answer display
function formatAnswer(answer) {
    if (Array.isArray(answer)) {
        return answer.join(', ');
    }
    return answer || '';
}

// Format content (supports HTML)
function formatContent(content) {
    if (typeof content === 'string') {
        // Simple line break handling
        return content.replace(/\n/g, '<br>');
    }
    return content;
}

// Handle reset
function handleReset() {
    loadQuestionData('data-before-submit.json');
    // Reset inputs
    document.querySelectorAll('input, textarea').forEach(input => {
        if (input.type === 'radio' || input.type === 'checkbox') {
            input.checked = false;
        } else {
            input.value = '';
        }
    });
    document.querySelectorAll('.radio-option, .checkbox-option').forEach(option => {
        option.classList.remove('checked');
    });
}

// Handle next question
function handleNext() {
    // Here you should load the next question's data
    // For demo purposes, we reload the before-submit state
    loadQuestionData('data-before-submit.json');
    switchToState('before');
}

// Handle review
function handleReview() {
    // Can expand more details or navigate to detail page
    console.log('View details');
}

// Handle retry
function handleRetry() {
    // Return to before-submit state
    loadQuestionData('data-before-submit.json');
    switchToState('before');
}

// Switch state
function switchToState(state) {
    currentState = state;
    document.querySelectorAll('.state-container').forEach(el => el.classList.add('hidden'));
    document.getElementById(`state-${state}`).classList.remove('hidden');
}

// Can directly set state and data via API (for backend calls)
window.setQuestionData = function(data) {
    currentQuestionData = data;
    if (data.state === 'before') {
        renderQuestion(data);
        switchToState('before');
    } else if (data.state === 'correct') {
        showCorrectState(data, data.userAnswer);
    } else if (data.state === 'wrong') {
        showWrongState(data, data.userAnswer);
    }
};

