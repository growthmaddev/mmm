import { Request, Response } from 'express';
import { compare, hash } from 'bcryptjs';
import { storage } from '../storage';
import { insertUserSchema } from '@shared/schema';
import { z } from 'zod';
import { validateRequest, AuthRequest } from '../middleware/auth';

// Login schema
const loginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8),
});

// Register schema with additional validation
const registerSchema = insertUserSchema.extend({
  email: z.string().email(),
  password: z.string().min(8),
  organizationName: z.string().min(2).optional(),
});

// Registration handler
export const register = async (req: Request, res: Response) => {
  try {
    const { username, email, password, firstName, lastName, organizationName } = registerSchema.parse(req.body);
    
    // Check if user already exists
    const existingUser = await storage.getUserByEmail(email);
    if (existingUser) {
      return res.status(409).json({ message: 'Email already registered' });
    }
    
    // Hash password
    const hashedPassword = await hash(password, 10);
    
    // Create organization if name provided
    let organizationId: number | undefined;
    if (organizationName) {
      const organization = await storage.createOrganization({ name: organizationName });
      organizationId = organization.id;
    }
    
    // Create user
    const user = await storage.createUser({
      username,
      email,
      password: hashedPassword,
      firstName,
      lastName,
      organizationId: organizationId,
      role: organizationId ? 'admin' : 'user' // First user in org is admin
    });
    
    // Create audit log
    await storage.createAuditLog({
      userId: user.id,
      organizationId: organizationId,
      action: 'user.register',
      details: { email: user.email },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'] || ''
    });
    
    // Create session
    req.session.userId = user.id;
    
    // Remove password from response
    const { password: _, ...userWithoutPassword } = user;
    
    return res.status(201).json({
      message: 'User registered successfully',
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Registration error:', error);
    return res.status(500).json({ message: 'Error during registration' });
  }
};

// Login handler
export const login = async (req: Request, res: Response) => {
  try {
    const { email, password } = loginSchema.parse(req.body);
    
    // Find user
    const user = await storage.getUserByEmail(email);
    if (!user) {
      return res.status(401).json({ message: 'Invalid email or password' });
    }
    
    // Check password
    const passwordValid = await compare(password, user.password);
    if (!passwordValid) {
      return res.status(401).json({ message: 'Invalid email or password' });
    }
    
    // Create session
    req.session.userId = user.id;
    
    // Create audit log
    await storage.createAuditLog({
      userId: user.id,
      organizationId: user.organizationId,
      action: 'user.login',
      details: { email: user.email },
      ipAddress: req.ip,
      userAgent: req.headers['user-agent'] || ''
    });
    
    // Remove password from response
    const { password: _, ...userWithoutPassword } = user;
    
    return res.json({
      message: 'Login successful',
      user: userWithoutPassword
    });
  } catch (error) {
    console.error('Login error:', error);
    return res.status(500).json({ message: 'Error during login' });
  }
};

// Logout handler
export const logout = (req: Request, res: Response) => {
  req.session.destroy(() => {
    res.clearCookie('connect.sid');
    return res.json({ message: 'Logout successful' });
  });
};

// Get current user
export const getCurrentUser = async (req: AuthRequest, res: Response) => {
  if (!req.user) {
    return res.status(401).json({ message: 'Not authenticated' });
  }
  
  try {
    const user = await storage.getUser(req.user.id);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }
    
    // Remove password from response
    const { password, ...userWithoutPassword } = user;
    
    return res.json(userWithoutPassword);
  } catch (error) {
    console.error('Get current user error:', error);
    return res.status(500).json({ message: 'Error fetching user data' });
  }
};

// Auth controller routes
export const authRoutes = {
  register: [validateRequest(registerSchema), register],
  login: [validateRequest(loginSchema), login],
  logout,
  getCurrentUser
};
