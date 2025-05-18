import { Request, Response, NextFunction } from "express";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import { db } from "../db";
import * as schema from "@shared/schema";
import { eq, and, gt } from "drizzle-orm";

// Extended Request type that includes the authenticated user ID
export interface AuthRequest extends Request {
  userId?: number;
}

// Handle validation errors gracefully
export const validateRequest = (schema: any) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const validationError = fromZodError(error);
        return res.status(400).json({ 
          message: "Validation error", 
          errors: validationError.details 
        });
      }
      next(error);
    }
  };
};

// Authentication middleware - verify token from cookies
export const isAuthenticated = async (req: AuthRequest, res: Response, next: NextFunction) => {
  try {
    const token = req.cookies?.auth_token;
    
    if (!token) {
      return res.status(401).json({ message: "Not authenticated" });
    }
    
    // Find valid session
    const [session] = await db
      .select()
      .from(schema.sessions)
      .where(
        and(
          eq(schema.sessions.token, token),
          gt(schema.sessions.expiresAt, new Date())
        )
      );
      
    if (!session) {
      return res.status(401).json({ message: "Session expired or invalid" });
    }
    
    // Add userId to request
    req.userId = session.userId;
    next();
  } catch (error) {
    console.error("Authentication error:", error);
    res.status(500).json({ message: "Authentication error" });
  }
};

// Create audit logs for important actions
export const auditLog = (action: string) => {
  return async (req: AuthRequest, res: Response, next: NextFunction) => {
    const originalEnd = res.end;
    res.end = function(this: Response, ...args: any[]) {
      // Only log successful requests
      if (res.statusCode >= 200 && res.statusCode < 300 && req.userId) {
        try {
          import("../storage").then(({ storage }) => {
            storage.createAuditLog({
              userId: req.userId!,
              action,
              details: {
                method: req.method,
                path: req.path,
                params: req.params,
                body: req.body
              },
              ipAddress: req.ip,
              userAgent: req.headers["user-agent"] || ""
            }).catch(err => console.error("Error creating audit log:", err));
          }).catch(err => console.error("Error importing storage:", err));
        } catch (error) {
          console.error("Error in audit logging:", error);
        }
      }
      return originalEnd.apply(this, args);
    };
    next();
  };
};