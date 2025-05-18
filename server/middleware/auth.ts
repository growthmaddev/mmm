import { Request, Response, NextFunction } from "express";
import { storage } from "../storage";
import { ZodError } from "zod";
import { formatValidationError } from "zod-validation-error";

export interface AuthRequest extends Request {
  user?: {
    id: number;
    username: string;
    email: string;
    organizationId: number | null;
    role: string;
  };
}

// Verifies that a user is authenticated
export const isAuthenticated = async (req: AuthRequest, res: Response, next: NextFunction) => {
  if (!req.session || !req.session.userId) {
    return res.status(401).json({ message: "Not authenticated" });
  }

  try {
    const user = await storage.getUser(req.session.userId);
    if (!user) {
      req.session.destroy(() => {});
      return res.status(401).json({ message: "User not found" });
    }

    // Set user information for route handlers
    req.user = {
      id: user.id,
      username: user.username,
      email: user.email,
      organizationId: user.organizationId,
      role: user.role
    };

    next();
  } catch (error) {
    console.error("Authentication middleware error:", error);
    return res.status(500).json({ message: "Server error during authentication" });
  }
};

// Verifies that a user is an admin
export const isAdmin = async (req: AuthRequest, res: Response, next: NextFunction) => {
  if (!req.user) {
    return res.status(401).json({ message: "Not authenticated" });
  }

  if (req.user.role !== "admin") {
    return res.status(403).json({ message: "Not authorized" });
  }

  next();
};

// Handle validation errors gracefully
export const validateRequest = (schema: any) => {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      schema.parse(req.body);
      next();
    } catch (error) {
      if (error instanceof ZodError) {
        const validationError = formatValidationError(error);
        return res.status(400).json({ 
          message: "Validation error", 
          errors: validationError.details 
        });
      }
      next(error);
    }
  };
};

// Create audit logs for important actions
export const auditLog = (action: string) => {
  return async (req: AuthRequest, res: Response, next: NextFunction) => {
    const originalEnd = res.end;
    res.end = function(this: Response, ...args: any[]) {
      // Only log successful requests
      if (res.statusCode >= 200 && res.statusCode < 300 && req.user) {
        try {
          storage.createAuditLog({
            userId: req.user.id,
            organizationId: req.user.organizationId || undefined,
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
        } catch (error) {
          console.error("Error in audit logging:", error);
        }
      }
      return originalEnd.apply(this, args);
    };
    next();
  };
};
