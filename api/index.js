const express = require("express");
const multer = require("multer");
const cors = require("cors");
const axios = require("axios");
const FormData = require("form-data");

const app = express();
const port = 3000;

// Enable CORS and JSON parsing
app.use(cors());
app.use(express.json());

// ✅ Use memory storage instead of disk (fixes Vercel error)
const upload = multer({ storage: multer.memoryStorage() });

app.get("/", (req, res) => {
    res.send("Hello, World!");
});

app.post("/vqa", upload.single("image"), async (req, res) => {
    if (!req.file || !req.body.question) {
        return res.status(400).json({ error: "Missing image or question" });
    }

    const imageBuffer = req.file.buffer;  // ✅ Store image in memory
    const userQuestion = req.body.question;

    try {
        // ✅ Step 1: Generate queries using DeepSeek V3
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
                'Authorization': 'Bearer hf_xxx', // 🔴 Replace with your actual token
                'Content-Type': 'application/json'
            }
        });

        const generatedQueries = deepSeekV3Response.data.choices[0].message.content.split('\n').filter(q => q.trim()).slice(0, 8);
        console.log("Generated Queries:", generatedQueries);

        // ✅ Step 2: Send Image & Queries to an External VQA API
        const vqaResults = {};
        for (const query of generatedQueries) {
            try {
                const vqaResponse = await processVQA(imageBuffer, query);
                console.log(`Query: ${query}, Answer: ${vqaResponse}`);
                vqaResults[query] = vqaResponse;
            } catch (error) {
                console.error(`Error processing query "${query}": ${error.message}`);
                return res.status(500).json({ error: "Internal Server Error" });
            }
        }

        // ✅ Step 3: Compile the answers using DeepSeek V3
        const deepSeekV3CompilePayload = {
            model: "deepseek-ai/DeepSeek-V3",
            messages: [
                {
                    role: "user",
                    content: `Compile the following queries and answers into a single sentence: ${JSON.stringify(vqaResults)}`
                }
            ],
            max_tokens: 500,
            stream: false
        };

        const deepSeekV3CompileResponse = await axios.post("https://router.huggingface.co/together/v1/chat/completions", deepSeekV3CompilePayload, {
            headers: {
                'Authorization': 'Bearer hf_xxx', // 🔴 Replace with your actual token
                'Content-Type': 'application/json'
            }
        });

        console.log("Compiled Response:", deepSeekV3CompileResponse.data.choices[0].message.content);

        // ✅ Step 4: Return the compiled response to the user
        res.json({ answer: deepSeekV3CompileResponse.data.choices[0].message.content });
    } catch (error) {
        console.error(`Error: ${error.message}`);
        res.status(500).json({ error: "Internal Server Error" });
    }
});

// ✅ Helper Function: Send image and question to a remote VQA API (instead of running Python)
async function processVQA(imageBuffer, question) {
    try {
        const formData = new FormData();
        formData.append("image", imageBuffer, { filename: "image.jpg", contentType: "image/jpeg" });
        formData.append("question", question);

        const vqaResponse = await axios.post("https://your-vqa-api.com/process", formData, {
            headers: { 
                ...formData.getHeaders(), 
                "Authorization": "Bearer YOUR_VQA_API_KEY" // 🔴 Replace with your API key
            }
        });

        return vqaResponse.data.answer;
    } catch (error) {
        throw new Error("Error processing VQA API");
    }
}

module.exports = app;
