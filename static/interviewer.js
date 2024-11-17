const video = document.getElementById('user-video');
const offvideo = document.getElementById('toggle-user-video');
const muteButton = document.getElementById('toggle-mute');
const transcribedTextDiv = document.getElementById('transcribed-text');

let videoOn = true; // Track the video state

function end() {
    window.location.href = "/templates/fedback.html";
}

function restart() {
    alert("You are redirecting to home page, restart the interview again");
    window.location.href = "/templates/home";
}

let timeLeft = 5 * 60; // 3 minutes
const timerElement = document.getElementById('timer');

const updateTimer = () => {
    const minutes = Math.floor(timeLeft / 60);
    const seconds = timeLeft % 60;
    timerElement.textContent = `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
    if (timeLeft > 0) {
        timeLeft--;
    } else {
        clearInterval(timerInterval);
        alert('Interview completed.');
        document.getElementById('feedback-form').style.display = 'block';
        window.location.href = "fedback.html";
    }
};

const timerInterval = setInterval(updateTimer, 1000);

function toggleVideo() {
    if (videoOn) {
        video.style.display = 'none'; // Hide the video
        offvideo.innerText = 'Turn On Video'; // Change button text
        videoOn = false; // Update the state
    } else {
        video.style.display = 'block'; // Show the video
        offvideo.innerText = 'Turn Off Video'; // Change button text
        videoOn = true; // Update the state
    }
}

// Add event listener for the mute button (optional functionality)
muteButton.addEventListener('click', () => {
    alert("Your microphone is in mute please unmute to speak...");
    // if (mute) {

    // } else {

    // }
});

async function fetchQuestion() {
    try {
        document.getElementById('spinner').style.display = 'block'; // Show spinner while fetching

        const response = await fetch('/get_question');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();

        if (data.question && data.question !== "") {
            // If a question is received, display it and hide the spinner
            document.getElementById('question-text').textContent = data.question;
            document.getElementById('spinner').style.display = 'none';
        } else {
            // If no question is available, retry fetching after 1 second
            setTimeout(fetchQuestion, 1000); // Retry every 1000ms (1 second)
        }
    } catch (error) {
        console.error("Error fetching question:", error);
        document.getElementById('question-text').textContent = "Error fetching question.";
        document.getElementById('spinner').style.display = 'none'; // Hide spinner on error
    }
}

// Call fetchQuestion when the page loads
window.onload = function() {
    fetchQuestion(); // Start fetching the question
};
