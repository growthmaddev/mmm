import { Request, Response } from 'express';
import multer from 'multer';
import fs from 'fs';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import { parse as csvParse } from 'csv-parse';
import { AuthRequest } from '../middleware/auth';

// Setup multer storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(import.meta.dirname, '..', '..', 'uploads');
    
    // Create directory if it doesn't exist
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueFilename = `${uuidv4()}-${file.originalname}`;
    cb(null, uniqueFilename);
  }
});

// File filter to only allow CSV and Excel files
const fileFilter = (req: Request, file: Express.Multer.File, cb: multer.FileFilterCallback) => {
  if (
    file.mimetype === 'text/csv' ||
    file.mimetype === 'application/vnd.ms-excel' ||
    file.mimetype === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
  ) {
    cb(null, true);
  } else {
    cb(new Error('Invalid file type. Only CSV and Excel files are allowed.'));
  }
};

// Configure multer
export const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024, // 50MB max file size
  }
});

// Function to validate CSV data
export const validateCsvData = async (filePath: string): Promise<{
  isValid: boolean;
  columns: string[];
  sampleData: any[];
  errors: string[];
}> => {
  return new Promise((resolve, reject) => {
    const errors: string[] = [];
    const parser = fs.createReadStream(filePath).pipe(
      csvParse({
        columns: true,
        skip_empty_lines: true,
        trim: true,
      })
    );

    const sampleData: any[] = [];
    let columns: string[] = [];
    
    parser.on('readable', function() {
      let record;
      while ((record = parser.read())) {
        sampleData.push(record);
        if (sampleData.length === 1) {
          columns = Object.keys(record);
        }
        if (sampleData.length >= 5) {
          break;
        }
      }
    });
    
    parser.on('error', function(err) {
      errors.push(`Error parsing CSV: ${err.message}`);
      resolve({ isValid: false, columns: [], sampleData: [], errors });
    });
    
    parser.on('end', function() {
      if (sampleData.length === 0) {
        errors.push('CSV file is empty or has no valid data rows');
        resolve({ isValid: false, columns: [], sampleData: [], errors });
        return;
      }
      
      // Check for date column
      const potentialDateColumns = columns.filter(col => 
        col.toLowerCase().includes('date') || 
        col.toLowerCase().includes('time') || 
        col.toLowerCase().includes('day')
      );
      
      if (potentialDateColumns.length === 0) {
        errors.push('No potential date column found. Please ensure your data includes a date column.');
      }
      
      // Check for potential metric columns
      const potentialMetricColumns = columns.filter(col => 
        sampleData.some(row => !isNaN(parseFloat(row[col])))
      );
      
      if (potentialMetricColumns.length === 0) {
        errors.push('No numeric metric columns found. Please ensure your data includes columns with numeric values.');
      }
      
      resolve({
        isValid: errors.length === 0,
        columns,
        sampleData,
        errors
      });
    });
  });
};

// Upload route handler
export const handleFileUpload = async (req: AuthRequest, res: Response) => {
  if (!req.file) {
    return res.status(400).json({
      message: 'No file uploaded',
    });
  }

  try {
    // Get project ID
    const projectId = req.body.projectId ? parseInt(req.body.projectId) : null;
    if (!projectId) {
      return res.status(400).json({ message: 'Project ID is required' });
    }

    // Validate the file
    const validationResult = await validateCsvData(req.file.path);
    
    // Return validation results
    return res.json({
      fileName: req.file.originalname,
      fileUrl: req.file.path,
      fileSize: req.file.size,
      validation: validationResult
    });
  } catch (error) {
    console.error('File upload error:', error);
    return res.status(500).json({
      message: 'Error processing the uploaded file',
      error: error.message,
    });
  }
};

// Get file download template
export const getFileTemplate = (req: Request, res: Response) => {
  const templateType = req.params.type;
  
  let templatePath;
  
  switch (templateType) {
    case 'marketing_data':
      templatePath = path.join(import.meta.dirname, '..', 'templates', 'marketing_data_template.csv');
      break;
    case 'channel_spend':
      templatePath = path.join(import.meta.dirname, '..', 'templates', 'channel_spend_template.csv');
      break;
    default:
      return res.status(400).json({ message: 'Invalid template type' });
  }
  
  if (!fs.existsSync(templatePath)) {
    // If the template doesn't exist, create a simple one
    let templateContent = '';
    
    if (templateType === 'marketing_data') {
      templateContent = 'Date,Sales,Channel1_Spend,Channel2_Spend,Channel3_Spend,Promotion,Holiday\n' +
        '2023-01-01,10000,1000,2000,3000,1,0\n' +
        '2023-01-02,12000,1500,2500,3500,1,0\n';
    } else if (templateType === 'channel_spend') {
      templateContent = 'Date,Channel,Spend,Impressions\n' +
        '2023-01-01,Search,1000,50000\n' +
        '2023-01-01,Display,2000,100000\n';
    }
    
    // Create directory if it doesn't exist
    const templateDir = path.dirname(templatePath);
    if (!fs.existsSync(templateDir)) {
      fs.mkdirSync(templateDir, { recursive: true });
    }
    
    fs.writeFileSync(templatePath, templateContent);
  }
  
  return res.download(templatePath);
};
