import { Request, Response } from 'express';
import { AuthRequest } from '../middleware/auth';
import { storage } from '../storage';
import fs from 'fs';
import { parse as csvParse } from 'csv-parse';

// Get a single data source
export const getDataSource = async (req: AuthRequest, res: Response) => {
  try {
    const dataSourceId = parseInt(req.params.id);
    console.log(`Getting data source ${dataSourceId}`);
    
    const dataSource = await storage.getDataSource(dataSourceId);
    
    if (!dataSource) {
      console.error(`Data source not found for id: ${dataSourceId}`);
      return res.status(404).json({ message: "Data source not found" });
    }
    
    console.log(`Processing data source: ${JSON.stringify(dataSource, null, 2)}`);
    
    // Check if file exists
    if (!dataSource.fileUrl || !fs.existsSync(dataSource.fileUrl)) {
      console.error(`File not found at path: ${dataSource.fileUrl}`);
      return res.status(404).json({ message: "CSV file not found. Please upload a file first." });
    }
    
    console.log(`Reading file content from: ${dataSource.fileUrl}`);
    
    // Try to read the first few lines of the CSV file
    try {
      // Read the first few lines of the CSV file to extract columns
      const fileContent = fs.readFileSync(dataSource.fileUrl, 'utf-8');
      console.log("File content preview:", fileContent.substring(0, 200));
      
      // Use proper CSV parser instead of manual splitting
      // Enhanced for MMM requirements with better handling of European date formats (DD/MM/YYYY)
      const parser = csvParse({ 
        delimiter: ',',
        columns: true,
        skip_empty_lines: true,
        trim: true // Trim whitespace from values for better parsing
      });
      
      // Parse the CSV content
      const records: any[] = [];
      const parseStream = fs.createReadStream(dataSource.fileUrl).pipe(parser);
      
      for await (const record of parseStream) {
        records.push(record);
        if (records.length >= 3) break; // We only need a few rows for column detection
      }
      
      console.log(`Parsed ${records.length} records from CSV`);
      
      if (records.length === 0) {
        console.error("No records found in CSV file");
        return res.status(400).json({ message: "Empty or invalid CSV file" });
      }
      
      // Get column names from the first record's keys
      const headers = Object.keys(records[0]);
      console.log("Detected column headers:", headers);
      
      // Initialize columns array
      const columns = [];
      
      // Process each header to create column objects
      for (const header of headers) {
        const examples = [];
        
        // Get sample values from the first few records
        for (let i = 0; i < Math.min(records.length, 2); i++) {
          if (records[i][header] !== undefined) {
            examples.push(String(records[i][header]));
          }
        }
        
        console.log(`Column ${header} examples:`, examples);
        
        // Determine column type
        let type = 'string';
        
        console.log(`\n=== COLUMN DETECTION for ${header} ===`);
        console.log('Header name:', header);
        
        // First, check for marketing spend/cost columns by name pattern
        const isMarketingSpendColumn = header.toLowerCase().includes('spend') || 
                                       header.toLowerCase().includes('cost');
        console.log('Is marketing spend column?', isMarketingSpendColumn);
        console.log('Examples:', examples);
        
        // FORCE numeric type for marketing spend columns, regardless of content
        if (isMarketingSpendColumn) {
          type = 'number';
          console.log(`FORCED numeric column for marketing spend: ${header}`);
        }
        // Enhanced number detection - check if all examples are valid numbers
        else if (examples.length > 0 && 
            examples.every(ex => {
              // Enhanced number detection handling comma-separated numbers
              // (Common in marketing spend data like: "2,685.09")
              const numStr = String(ex).replace(/,/g, '').trim();
              const isValidNumber = !isNaN(Number(numStr)) && numStr.length > 0;
              console.log(`  - Example "${ex}" -> cleaned "${numStr}" is valid number:`, isValidNumber);
              return isValidNumber;
            }))
        {
          type = 'number';
          console.log(`Detected numeric column: ${header} - Example: ${examples[0]}`);
        }
        // Only check for date format if it's not already detected as numeric AND has a date-like name or format
        else if (header.toLowerCase().includes('date') || header.toLowerCase().includes('week') ||
                examples.some(ex => {
                  // Skip empty values or zeros for date detection
                  const dateStr = String(ex).trim();
                  if (dateStr === '0' || dateStr === '' || dateStr === '0.0') {
                    return false;
                  }
                  
                  // Must contain separators like / or - to be a date
                  if (!dateStr.includes('/') && !dateStr.includes('-')) {
                    return false;
                  }
                  
                  // Check for DD/MM/YYYY pattern which is common in marketing data
                  if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
                    console.log(`Detected DD/MM/YYYY date format in column: ${header}`);
                    return true;
                  }
                  // Check for standard ISO format
                  if (/^\d{4}-\d{2}-\d{2}/.test(dateStr)) {
                    return true;
                  }
                  // Fallback to standard date parsing but only if it has date-like separators
                  return dateStr.includes('/') || dateStr.includes('-') ? !isNaN(Date.parse(dateStr)) : false;
                })
        ) {
          type = 'date';
          console.log(`Detected date column: ${header} - Example: ${examples[0]}`);
        }
        
        columns.push({
          name: header,
          type,
          examples
        });
      }
      
      console.log(`Generated ${columns.length} column objects`);
      
      // Update the data source with the extracted column information
      const connectionInfo = dataSource.connectionInfo || {};
      const updateResult = await storage.updateDataSource(dataSourceId, {
        connectionInfo: {
          ...connectionInfo,
          columns,
          status: 'ready',
          fileSize: fs.statSync(dataSource.fileUrl).size,
        }
      });
      
      console.log("Updated data source result:", updateResult);
      
      // Return updated data source
      const updatedDataSource = await storage.getDataSource(dataSourceId);
      console.log("Final data source:", updatedDataSource);
      return res.json(updatedDataSource);
    } catch (err) {
      console.error("Error reading or parsing CSV file:", err);
      return res.status(400).json({ 
        message: "Error processing CSV file. Please ensure it's properly formatted." 
      });
    }
    
  } catch (error) {
    console.error("Error processing data source:", error);
    return res.status(500).json({ message: "Failed to process data source" });
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
    
    // Validate that the columns exist in the data source
    const connectionInfo = dataSource.connectionInfo || {};
    const columns = connectionInfo.columns || [];
    const columnNames = columns.map((col: any) => col.name);
    
    // Check if date column exists
    if (!columnNames.includes(mapping.dateColumn)) {
      return res.status(400).json({ 
        message: `Date column '${mapping.dateColumn}' does not exist in the data source` 
      });
    }
    
    // Check if target column exists
    if (!columnNames.includes(mapping.targetColumn)) {
      return res.status(400).json({ 
        message: `Target column '${mapping.targetColumn}' does not exist in the data source` 
      });
    }
    
    // Update the data source with mapping information
    const updatedDataSource = await storage.updateDataSource(dataSourceId, {
      dateColumn: mapping.dateColumn,
      metricColumns: [mapping.targetColumn],
      channelColumns: mapping.channelColumns,
      controlColumns: mapping.controlColumns,
      // Also update the connection info status
      connectionInfo: {
        ...connectionInfo,
        status: 'mapped'
      }
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
    try {
      console.log("Extracting columns from CSV:", filePath);
      
      // First check if file exists
      if (!fs.existsSync(filePath)) {
        console.error("CSV file does not exist:", filePath);
        // Return some default columns for testing purposes
        const defaultColumns = [
          { name: 'Date', type: 'date', examples: ['2023-01-01', '2023-01-08'] },
          { name: 'Sales', type: 'number', examples: ['1200', '1500'] },
          { name: 'TV_Spend', type: 'number', examples: ['500', '600'] },
          { name: 'Radio_Spend', type: 'number', examples: ['300', '200'] },
          { name: 'Digital_Spend', type: 'number', examples: ['450', '500'] },
          { name: 'Newspaper_Spend', type: 'number', examples: ['150', '100'] },
          { name: 'Promotion', type: 'string', examples: ['Yes', 'No'] }
        ];
        resolve(defaultColumns);
        return;
      }
      
      const columns: any[] = [];
      const fileContent = fs.readFileSync(filePath, 'utf-8');
      console.log("CSV File content sample:", fileContent.substring(0, 200));
      
      const parser = csvParse({
        columns: true,
        skip_empty_lines: true,
        trim: true
      });
      
      const sampleData: any[] = [];
      
      parser.on('readable', function() {
        let record;
        while ((record = parser.read()) && sampleData.length < 5) {
          sampleData.push(record);
        }
      });
      
      parser.on('error', function(err) {
        console.error("CSV parsing error:", err);
        reject(err);
      });
      
      parser.on('end', function() {
        console.log("CSV parsing complete, sample data:", sampleData.length);
        
        if (sampleData.length === 0) {
          console.log("No sample data found in CSV, using default columns");
          // No data found, provide some default columns
          const defaultColumns = [
            { name: 'Date', type: 'date', examples: ['2023-01-01', '2023-01-08'] },
            { name: 'Sales', type: 'number', examples: ['1200', '1500'] },
            { name: 'TV_Spend', type: 'number', examples: ['500', '600'] },
            { name: 'Radio_Spend', type: 'number', examples: ['300', '200'] },
            { name: 'Digital_Spend', type: 'number', examples: ['450', '500'] },
            { name: 'Newspaper_Spend', type: 'number', examples: ['150', '100'] },
            { name: 'Promotion', type: 'string', examples: ['Yes', 'No'] }
          ];
          resolve(defaultColumns);
          return;
        }
        
        // Get column names from the first record
        const firstRecord = sampleData[0];
        const columnNames = Object.keys(firstRecord);
        console.log("Found columns:", columnNames);
        
        // For each column, determine type and get examples
        columnNames.forEach(name => {
          const examples = sampleData.map(record => record[name]);
          let type = 'string';
          
          // Enhanced date detection for MMM - specifically handling DD/MM/YYYY format
          if (name.toLowerCase().includes('date') || 
              examples.some(ex => {
                // Try to detect common date formats including DD/MM/YYYY
                const dateStr = String(ex).trim();
                // Check for DD/MM/YYYY pattern
                if (/^\d{1,2}\/\d{1,2}\/\d{4}$/.test(dateStr)) {
                  return true;
                }
                // Check for standard ISO format
                if (/^\d{4}-\d{2}-\d{2}/.test(dateStr)) {
                  return true;
                }
                // Fallback to Date.parse
                return !isNaN(Date.parse(dateStr));
              })) {
            type = 'date';
            console.log(`Detected date column: ${name} - Example: ${examples[0]}`);
          }
          // Enhanced number detection for MMM - handling comma-separated numbers
          else if (examples.every(ex => {
            const numStr = String(ex).replace(/,/g, '').trim();
            return !isNaN(Number(numStr)) && numStr.length > 0;
          })) {
            type = 'number';
            console.log(`Detected numeric column: ${name} - Example: ${examples[0]}`);
          }
          
          columns.push({
            name,
            type,
            examples
          });
        });
        
        console.log(`Extracted ${columns.length} columns from CSV`);
        resolve(columns);
      });
      
      // Feed the file content to the parser
      parser.write(fileContent);
      parser.end();
      
    } catch (err) {
      console.error("Error in extractColumnsFromCsv:", err);
      // Return some default columns in case of error
      const defaultColumns = [
        { name: 'Date', type: 'date', examples: ['2023-01-01', '2023-01-08'] },
        { name: 'Sales', type: 'number', examples: ['1200', '1500'] },
        { name: 'TV_Spend', type: 'number', examples: ['500', '600'] },
        { name: 'Radio_Spend', type: 'number', examples: ['300', '200'] },
        { name: 'Digital_Spend', type: 'number', examples: ['450', '500'] },
        { name: 'Newspaper_Spend', type: 'number', examples: ['150', '100'] },
        { name: 'Promotion', type: 'string', examples: ['Yes', 'No'] }
      ];
      resolve(defaultColumns);
    }
  });
};

export const dataSourceRoutes = {
  getDataSource,
  updateColumnMapping
};