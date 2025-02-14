const express = require("express");
const multer = require("multer");
const cors = require("cors");
const { exec } = require("child_process");
const path = require("path");
const axios = require("axios");

const app = express();
const port = 3000;

// Enable CORS for frontend requests
app.use(cors());
app.use(express.json());

// Configure multer for image uploads
const upload = multer({ dest: "uploads/" });

app.get("/", (req, res) => {
    res.send("Hello, World!"); 
});

app.post("/vqa", upload.single("image"), async (req, res) => {
    if (!req.file || !req.body.question) {
        return res.status(400).json({ error: "Missing image or question" });
    }

    const imagePath = req.file.path;  // Path to the uploaded image
    const userQuestion = req.body.question;  // User's question

    try {
        // Step 1: Generate one-line queries from the user's question using DeepSeek V3
        const deepSeekV3Payload = {
            model: "deepseek-ai/DeepSeek-V3",
            messages: [
                {
                    role: "user",
                    content: `Generate up to 8 queries from the following question: ${userQuestion}`
                }
            ],
            max_tokens: 500,
            stream: false
        };

        const deepSeekV3Response = await axios.post("https://router.huggingface.co/together/v1/chat/completions", deepSeekV3Payload, {
            headers: {
                'Authorization': 'Bearer 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', // Replace with your actual token
                'Content-Type': 'application/json'
            }
        });

        const generatedQueries = deepSeekV3Response.data.choices[0].message.content.split('\n').filter(query => query.trim()).slice(0, 8);
        console.log("Generated Queries:", generatedQueries);

        // Step 2: Ask the generated queries to the Python script one by one and collect the answers
        const results = {};

        for (const query of generatedQueries) {
            try {
                const answer = await askQuestion(imagePath, query);
                console.log(`Query: ${query}, Answer: ${answer}`);
                results[query] = answer;
            } catch (error) {
                console.error(`Error processing query "${query}": ${error.message}`);
                return res.status(500).json({ error: "Internal Server Error" });
            }
        }

        // Step 3: Compile the answers into a single response using DeepSeek V3
        const deepSeekV3CompilePayload = {
            model: "deepseek-ai/DeepSeek-V3",
            messages: [
                {
                    role: "user",
                    content: `Compile the following queries and answers into a single sentence: ${JSON.stringify(results)}`
                }
            ],
            max_tokens: 500,
            stream: false
        };

        const deepSeekV3CompileResponse = await axios.post("https://router.huggingface.co/together/v1/chat/completions", deepSeekV3CompilePayload, {
            headers: {
                'Authorization': 'Bearer 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', // Replace with your actual token
                'Content-Type': 'application/json'
            }
        });

        console.log("Compiled Response:", deepSeekV3CompileResponse.data.choices[0].message.content);

        // Step 4: Return the compiled response to the user
        res.json(deepSeekV3CompileResponse.data.choices[0].message.content);
    } catch (error) {
        console.error(`Error: ${error.message}`);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

function askQuestion(imagePath, question) {
    return new Promise((resolve, reject) => {
        exec(`set HF_HUB_DISABLE_SYMLINKS_WARNING=1 && python vqa.py ${imagePath} "${question}"`, (error, stdout, stderr) => {
            if (error) {
                return reject(error);
            }
            if (stderr) {
                console.error(`stderr: ${stderr}`);
            }

            try {
                const response = JSON.parse(stdout);
                resolve(response.answer);
            } catch (parseError) {
                reject(parseError);
            }
        });
    });
}

app.listen(port, () => {
    console.log(`Server running on http://localhost:${port}`);
});