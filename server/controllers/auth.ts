import { Request, Response } from "express";
import { db } from "../db";
import * as schema from "@shared/schema";
import { storage } from "../storage";
import bcrypt from "bcryptjs";
import { eq } from "drizzle-orm";
import { v4 as uuidv4 } from "uuid";
import { AuthRequest } from "../middleware/auth";

// Register a new user
export const register = async (req: Request, res: Response) => {
  try {
    const { email, password, username, firstName, lastName } = schema.registerUserSchema.parse(req.body);
    
    // Check if email already exists
    const existingUser = await storage.getUserByEmail(email);
    if (existingUser) {
      return res.status(400).json({ message: "Email already in use" });
    }
    
    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    // Create user
    const user = await storage.createUser({
      email,
      password: hashedPassword,
      username,
      firstName,
      lastName
    });
    
    // Create a session for the user
    const token = uuidv4();
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7); // 7 days from now
    
    await db.insert(schema.sessions).values({
      userId: user.id,
      token,
      expiresAt
    });
    
    // Set token as cookie
    res.cookie("auth_token", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    });
    
    // Return user info (without password)
    const { password: _, ...userWithoutPassword } = user;
    res.status(201).json(userWithoutPassword);
    
  } catch (error) {
    console.error("Registration error:", error);
    res.status(500).json({ message: "Failed to register user" });
  }
};

// Login user
export const login = async (req: Request, res: Response) => {
  try {
    const { email, password } = schema.loginUserSchema.parse(req.body);
    
    // TESTING ONLY: Accept any credentials
    console.log("Test mode: Accepting any login credentials");
    
    // Create a mock user for testing
    const mockUser = {
      id: 1,
      email: email || "test@example.com",
      firstName: "Test",
      lastName: "User",
      username: "testuser",
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    // Set a test token cookie
    const token = "test-token-" + Date.now();
    res.cookie("auth_token", token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    });
    
    // Return mock user
    res.json(mockUser);
    
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Failed to login" });
  }
};

// Logout
export const logout = (req: Request, res: Response) => {
  try {
    // Clear auth cookie
    res.clearCookie("auth_token");
    res.json({ message: "Logged out successfully" });
  } catch (error) {
    console.error("Logout error:", error);
    res.status(500).json({ message: "Failed to logout" });
  }
};

// Get current user
export const getCurrentUser = async (req: AuthRequest, res: Response) => {
  try {
    // TESTING ONLY: Always return a mock user
    console.log("Test mode: Returning mock user for getCurrentUser");
    
    // Create a mock user for testing
    const mockUser = {
      id: 1,
      email: "test@example.com",
      firstName: "Test",
      lastName: "User",
      username: "testuser",
      createdAt: new Date(),
      updatedAt: new Date()
    };
    
    res.json(mockUser);
  } catch (error) {
    console.error("Get current user error:", error);
    res.status(500).json({ message: "Failed to get current user" });
  }
};

export const authRoutes = {
  register,
  login,
  logout,
  getCurrentUser
};