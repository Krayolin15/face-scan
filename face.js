// Step 1: Select elements
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectBtn = document.getElementById('detect-btn');
    const moodResult = document.getElementById('mood-result');

    // Step 2: Request webcam access and stream to video element
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
      } catch (err) {
        console.error('Camera access error:', err);
        moodResult.textContent = 'Unable to access camera.';
      }
    }

    // Start camera on page load
    startCamera();

    // Step 3: Capture current frame from video and send to backend
    async function detectMood() {
      // Draw current video frame on hidden canvas
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Convert canvas image to base64 JPEG
      const imageData = canvas.toDataURL('image/jpeg');

      moodResult.textContent = 'Analyzing mood...';

      try {
        // Send image data to backend for analysis
        const response = await fetch('http://127.0.0.1:5000/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData }),
        });

        const data = await response.json();

        if (data.mood) {
          moodResult.textContent = `Mood detected: ${data.mood.toUpperCase()}`;
        } else if (data.error) {
          moodResult.textContent = `Error: ${data.error}`;
        } else {
          moodResult.textContent = 'Unexpected response from server.';
        }
      } catch (error) {
        console.error('Fetch error:', error);
        moodResult.textContent = 'Error communicating with server.';
      }
    }

    // Step 4: Attach event listener to button
    detectBtn.addEventListener('click', detectMood);