require("dotenv").config({ path: require("path").resolve(__dirname, "../.env") });
const express = require("express");
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const cors = require("cors");
const dns = require("dns");
const { createProxyMiddleware } = require("http-proxy-middleware");

// 🔧 Force Google DNS to resolve MongoDB Atlas SRV records
dns.setServers(["8.8.8.8", "8.8.4.4"]);

const app = express();

// 🚀 Proxy Machine Learning endpoints directly to the Python Render Service
const pythonApiUrl = process.env.PYTHON_API_URL || "https://smart-agriculture-system-u0tq.onrender.com";
app.use(['/predict_crop', '/predict_fertilizer', '/predict_yield'], createProxyMiddleware({
  target: pythonApiUrl,
  changeOrigin: true,
}));

app.use(express.json());
app.use(cors());
app.use(express.static("frontend"));

// 🔗 MongoDB Atlas Connection
mongoose.connect(process.env.MONGODB_URI)
  .then(() => console.log("MongoDB Connected ✅"))
  .catch(err => console.log("MongoDB Connection Error:", err));

// 🔹 User Schema
const UserSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  region: {
    type: String,
    required: true
  },
  soil: {
    type: String,
    required: true
  },
  password: {
    type: String,
    required: true
  },
  phone: { type: String, default: "" },
  language: { type: String, default: "en" },
  landArea: { type: Number, default: 0 },
  primaryCrop: { type: String, default: "" },
  irrigationType: { type: String, default: "Tube Well" },
  farmingStyle: { type: String, default: "Conventional" },
  history: [{
    analysisType: String,
    result: String,
    date: { type: Date, default: Date.now }
  }]
}, { timestamps: true });

const User = mongoose.model("User", UserSchema);

// ================= LOGIN =================
app.post("/login", async (req, res) => {
  try {
    const { email, password } = req.body;

    // Validation
    if (!email || !password) {
      return res.status(400).json({ success: false, message: "All fields required" });
    }

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({
        success: false,
        message: "User not found. Please register first."
      });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ success: false, message: "Incorrect password" });
    }

    res.json({ success: true, message: "Login successful 🎉", name: user.name, email: user.email, region: user.region, soil: user.soil });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ success: false, message: "Server error. Please try again." });
  }
});

// ================= REGISTER =================
app.post("/register", async (req, res) => {
  try {
    const { name, email, region, soil, password } = req.body;

    // Validation
    if (!name || !email || !region || !soil || !password) {
      return res.status(400).json({ success: false, message: "All fields required" });
    }

    // Email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({ success: false, message: "Invalid email format" });
    }

    if (password.length < 6) {
      return res.status(400).json({
        success: false,
        message: "Password must be at least 6 characters"
      });
    }

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(409).json({
        success: false,
        message: "User already exists. Please login."
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);

    const newUser = new User({
      name,
      email,
      region,
      soil,
      password: hashedPassword
    });

    await newUser.save();
    res.status(201).json({ success: true, message: "Registration successful ✅" });
  } catch (error) {
    console.error("Registration error:", error);
    res.status(500).json({ success: false, message: "Server error. Please try again." });
  }
});

// ================= PROFILE Endpoints =================
app.get("/api/profile/:email", async (req, res) => {
  try {
    const user = await User.findOne({ email: req.params.email }).select("-password");
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }
    res.json({ success: true, user });
  } catch (error) {
    console.error("Fetch profile error:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

app.put("/api/profile", async (req, res) => {
  try {
    const { email, name, phone, language, region, landArea, soil, primaryCrop, irrigationType, farmingStyle } = req.body;

    if (!email) {
      return res.status(400).json({ success: false, message: "Email is required to update profile" });
    }

    const updatedUser = await User.findOneAndUpdate(
      { email },
      { name, phone, language, region, landArea, soil, primaryCrop, irrigationType, farmingStyle },
      { new: true }
    ).select("-password");

    if (!updatedUser) {
      return res.status(404).json({ success: false, message: "User not found" });
    }

    res.json({ success: true, message: "Profile updated successfully", user: updatedUser });
  } catch (error) {
    console.error("Update profile error:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// ================= HISTORY Endpoints =================
app.post("/api/history", async (req, res) => {
  try {
    const { email, analysisType, result } = req.body;
    if (!email || !analysisType || !result) {
      return res.status(400).json({ success: false, message: "Missing required fields" });
    }
    const user = await User.findOneAndUpdate(
      { email },
      { $push: { history: { analysisType, result, date: new Date() } } },
      { new: true }
    );
    if (!user) {
      return res.status(404).json({ success: false, message: "User not found" });
    }
    res.json({ success: true, message: "History saved successfully" });
  } catch (error) {
    console.error("Save history error:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
