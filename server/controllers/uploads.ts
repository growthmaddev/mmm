import { Request, Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { v4 as uuidv4 } from 'uuid';
import { storage } from '../storage';

// Setup multer storage
const uploadDir = path.join(process.cwd(), 'uploads');
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const fileStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueFilename = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueFilename);
  }
});

// Configure multer
export const uploadMiddleware = multer({
  storage: fileStorage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'text/csv' || 
        file.mimetype === 'application/vnd.ms-excel' ||
        file.mimetype === 'application/csv') {
      cb(null, true);
    } else {
      cb(new Error('Only CSV files are allowed'));
    }
  }
});

// File upload controller
export const uploadFile = async (req: AuthRequest, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    if (!req.body.projectId) {
      return res.status(400).json({ error: 'Project ID is required' });
    }

    const projectId = parseInt(req.body.projectId);
    
    // Create a data source entry in the database
    const dataSource = await storage.createDataSource({
      projectId,
      type: 'csv_upload',
      fileName: req.file.originalname,
      fileUrl: req.file.path,
      connectionInfo: {
        status: 'ready',
        columns: [],
        fileSize: req.file.size
      },
      createdById: req.userId || 1
    });

    return res.status(200).json({
      success: true,
      message: 'File uploaded successfully',
      dataSource: {
        id: dataSource.id,
        fileName: req.file.originalname,
        fileSize: req.file.size
      }
    });
  } catch (error) {
    console.error('Upload error:', error);
    return res.status(500).json({ 
      error: 'Server error during file upload',
      message: error instanceof Error ? error.message : 'Unknown error' 
    });
  }
};