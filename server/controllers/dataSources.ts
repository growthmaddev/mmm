import { Request, Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { storage } from '../storage';
import fs from 'fs';
import { parse as csvParse } from 'csv-parse';

// Get a single data source
export const getDataSource = async (req: AuthRequest, res: Response) => {
  try {
    const dataSourceId = parseInt(req.params.id);
    const dataSource = await storage.getDataSource(dataSourceId);
    
    if (!dataSource) {
      return res.status(404).json({ message: "Data source not found" });
    }
    
    // If columns haven't been extracted yet, try to extract them
    if (!dataSource.connectionInfo?.columns || dataSource.connectionInfo.columns.length === 0) {
      if (dataSource.fileUrl && fs.existsSync(dataSource.fileUrl)) {
        const columns = await extractColumnsFromCsv(dataSource.fileUrl);
        
        // Update the data source with column information
        await storage.updateDataSource(dataSourceId, {
          connectionInfo: {
            ...dataSource.connectionInfo,
            columns
          }
        });
        
        // Return updated data source
        const updatedDataSource = await storage.getDataSource(dataSourceId);
        return res.json(updatedDataSource);
      }
    }
    
    return res.json(dataSource);
  } catch (error) {
    console.error("Error fetching data source:", error);
    return res.status(500).json({ message: "Failed to fetch data source" });
  }
};

// Update column mapping for a data source
export const updateColumnMapping = async (req: AuthRequest, res: Response) => {
  try {
    const dataSourceId = parseInt(req.params.id);
    const mapping = req.body;
    
    if (!mapping) {
      return res.status(400).json({ message: "No mapping data provided" });
    }
    
    const dataSource = await storage.getDataSource(dataSourceId);
    
    if (!dataSource) {
      return res.status(404).json({ message: "Data source not found" });
    }
    
    // Update the data source with mapping information
    const updatedDataSource = await storage.updateDataSource(dataSourceId, {
      dateColumn: mapping.dateColumn,
      metricColumns: [mapping.targetColumn],
      channelColumns: mapping.channelColumns,
      controlColumns: mapping.controlColumns
    });
    
    return res.json(updatedDataSource);
  } catch (error) {
    console.error("Error updating column mapping:", error);
    return res.status(500).json({ message: "Failed to update column mapping" });
  }
};

// Helper function to extract columns from CSV file
const extractColumnsFromCsv = async (filePath: string): Promise<any[]> => {
  return new Promise((resolve, reject) => {
    const columns: any[] = [];
    const parser = fs.createReadStream(filePath).pipe(
      csvParse({
        columns: true,
        skip_empty_lines: true,
        trim: true,
      })
    );
    
    const sampleData: any[] = [];
    
    parser.on('readable', function() {
      let record;
      while ((record = parser.read()) && sampleData.length < 5) {
        sampleData.push(record);
      }
    });
    
    parser.on('error', function(err) {
      reject(err);
    });
    
    parser.on('end', function() {
      if (sampleData.length === 0) {
        resolve([]);
        return;
      }
      
      // Get column names from the first record
      const firstRecord = sampleData[0];
      const columnNames = Object.keys(firstRecord);
      
      // For each column, determine type and get examples
      columnNames.forEach(name => {
        const examples = sampleData.map(record => record[name]);
        let type = 'string';
        
        // Try to determine if it's a date
        if (examples.some(ex => !isNaN(Date.parse(String(ex))))) {
          type = 'date';
        }
        // Try to determine if it's a number
        else if (examples.every(ex => !isNaN(Number(ex)))) {
          type = 'number';
        }
        
        columns.push({
          name,
          type,
          examples
        });
      });
      
      resolve(columns);
    });
  });
};

export const dataSourceRoutes = {
  getDataSource,
  updateColumnMapping
};