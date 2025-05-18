import { Request, Response, NextFunction } from "express";
import { ZodError } from "zod";
import { fromZodError } from "zod-validation-error";
import { User } from "@shared/schema";

// Extended Request type that includes the authenticated user
export interface AuthRequest extends Request {
  user?: any;
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

// Create audit logs for important actions
export const auditLog = (action: string) => {
  return async (req: AuthRequest, res: Response, next: NextFunction) => {
    const originalEnd = res.end;
    res.end = function(this: Response, ...args: any[]) {
      // Only log successful requests
      if (res.statusCode >= 200 && res.statusCode < 300 && req.user) {
        try {
          const userId = req.user.claims?.sub;
          if (userId) {
            import("../storage").then(({ storage }) => {
              storage.createAuditLog({
                userId,
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
            }).catch(err => console.error("Error importing storage:", err));
          }
        } catch (error) {
          console.error("Error in audit logging:", error);
        }
      }
      return originalEnd.apply(this, args);
    };
    next();
  };
};