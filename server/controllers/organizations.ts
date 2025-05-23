import { Request, Response } from "express";
import { storage } from "../storage";
import { AuthRequest } from "../middleware/auth";
import * as schema from "@shared/schema";
import { eq } from "drizzle-orm";
import { db } from "../db";

// Create a new organization and link it to the user
export const createOrganization = async (req: AuthRequest, res: Response) => {
  try {
    if (!req.userId) {
      return res.status(401).json({ message: "Not authenticated" });
    }
    
    const user = await storage.getUser(req.userId);
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    
    if (user.organizationId) {
      return res.status(400).json({ message: "User already belongs to an organization" });
    }
    
    // Create the organization
    const organization = await storage.createOrganization({
      name: req.body.name
    });
    
    // Link user to the organization
    await db.update(schema.users)
      .set({ organizationId: organization.id })
      .where(eq(schema.users.id, user.id));
    
    // Get updated user
    const updatedUser = await storage.getUser(req.userId);
    
    return res.status(201).json({
      organization,
      user: {
        ...updatedUser,
        password: undefined
      }
    });
  } catch (error) {
    console.error("Error creating organization:", error);
    return res.status(500).json({ message: "Failed to create organization" });
  }
};

// Get the current user's organization
export const getUserOrganization = async (req: AuthRequest, res: Response) => {
  try {
    if (!req.userId) {
      return res.status(401).json({ message: "Not authenticated" });
    }
    
    const user = await storage.getUser(req.userId);
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }
    
    if (!user.organizationId) {
      return res.status(404).json({ message: "User does not belong to an organization" });
    }
    
    const organization = await storage.getOrganization(user.organizationId);
    if (!organization) {
      return res.status(404).json({ message: "Organization not found" });
    }
    
    return res.json(organization);
  } catch (error) {
    console.error("Error fetching organization:", error);
    return res.status(500).json({ message: "Failed to fetch organization" });
  }
};

export const organizationRoutes = {
  createOrganization,
  getUserOrganization
};