# MarketMixMaster End-to-End Testing Checklist

This document provides a comprehensive testing checklist for verifying the complete workflow of MarketMixMaster with the new MMM fixed parameter implementation.

## Environment Setup

- [ ] Application is running on the correct port
- [ ] Database connection is established (check console logs)
- [ ] Required Python environment is available with all dependencies

## 1. Project Creation

- [ ] Navigate to the home page
- [ ] Click "Create New Project" button
- [ ] Enter project details:
  - [ ] Project Name: "E2E Test Project"
  - [ ] Description: "End-to-end testing of MMM implementation"
- [ ] Click "Create Project" button
- [ ] Verify project appears in the projects list
- [ ] Verify redirection to the project dashboard

**Network/Console Checks:**
- [ ] Verify POST request to `/api/projects` with correct payload
- [ ] Verify 200 OK response with project ID

## 2. Data Upload

### Test Data Requirements

Prepare a CSV file with the following characteristics:
- [ ] Date column in DD/MM/YYYY format (e.g., "01/05/2023")
- [ ] At least 3 marketing channel spend columns with comma-separated numeric values (e.g., "2,685.09")
- [ ] 1 response/target column (e.g., Sales or Revenue)
- [ ] At least 8 weeks of continuous data (minimum 8 rows)
- [ ] Optional control variables (Temperature, Holidays, Promotions)

### Upload Steps

- [ ] Navigate to the Data Upload page
- [ ] Click "Upload Data" button
- [ ] Select prepared CSV file
- [ ] Verify preview shows correctly with all columns
- [ ] Click "Upload" button
- [ ] Verify success message
- [ ] Verify file appears in data sources list

**Network/Console Checks:**
- [ ] Verify file upload POST request to `/api/projects/{id}/data-sources`
- [ ] Verify server logs show successful file parsing
- [ ] Check for any warnings about data format in console logs

## 3. Column Mapping

- [ ] Navigate to Column Mapping page
- [ ] Verify data preview table displays correctly

### Date Column Mapping
- [ ] Select date column from dropdown
- [ ] Verify date values are recognized correctly (check format: DD/MM/YYYY)

### Channel Column Mapping
- [ ] Check/select all marketing spend columns
- [ ] Verify numeric values display correctly (comma separators should be properly recognized)
- [ ] Optionally rename channels to friendly names

### Target Column Mapping
- [ ] Select response/target column (e.g., Sales, Revenue)

### Control Variables Mapping
- [ ] Map any control variables if available (Temperature, Holidays, Promotions)

- [ ] Click "Save Mapping" button
- [ ] Verify success message
- [ ] Verify redirection to the next step

**Network/Console Checks:**
- [ ] Verify POST request to `/api/projects/{id}/data-sources/{dataSourceId}/mapping`
- [ ] Verify console logs show `Converting data in DD/MM/YYYY format` (if applicable)
- [ ] Verify console logs show `Handling comma-separated numbers` (if applicable)

## 4. Model Setup

### Basic Configuration
- [ ] Navigate to Model Setup page
- [ ] Set model name: "E2E Test Model"
- [ ] Verify target variable is pre-selected based on column mapping
- [ ] Toggle AI assistant for intelligent defaults (if applicable)

### Adstock Settings
- [ ] Navigate to Adstock tab
- [ ] Verify all channels are listed
- [ ] Set appropriate adstock values for each channel (1-4 weeks)

### Saturation Settings
- [ ] Navigate to Saturation tab
- [ ] Verify enhanced UI showing L, k, x0 parameters for each channel
- [ ] Set the following parameter values for a test channel:
  - [ ] L (Max Effect): 1.0
  - [ ] k (Slope): 0.0001
  - [ ] x0 (Inflection): 50000
- [ ] Verify tooltips explain the parameters

### Control Variables
- [ ] Navigate to Control Variables tab
- [ ] Toggle appropriate control variables

- [ ] Click "Create Model" button
- [ ] Verify success message
- [ ] Verify model appears in the existing models list

**Network/Console Checks:**
- [ ] Verify POST request to `/api/models` with correct configuration
- [ ] Check that saturation parameters (L, k, x0) are included in the payload
- [ ] Verify 200 OK response with model ID

## 5. Model Training

- [ ] In the existing models section, find the created model
- [ ] Click "Start Training" button
- [ ] Verify training status changes to "training"
- [ ] Verify progress indicator appears
- [ ] Wait for training to complete (may take a few minutes)

**Network/Console Checks:**
- [ ] Verify POST request to `/api/models/{id}/train`
- [ ] Check server logs for:
  - [ ] `Using fixed parameter MMM implementation`
  - [ ] `fit_mmm_fixed_params.py is being called`
  - [ ] `Applying TensorVariable patch`
- [ ] Verify results transformation logs

## 6. Results Page

- [ ] Navigate to (or get redirected to) Results page
- [ ] Verify model appears with completion status

### Overview Tab
- [ ] Verify key metrics are displayed:
  - [ ] Model accuracy percentage
  - [ ] Top performing channel
  - [ ] ROI range
  - [ ] Recommendations

### Sales Decomposition Tab
- [ ] Verify sales decomposition chart loads
- [ ] Check baseline (intercept) percentage
- [ ] Verify each channel's contribution is displayed
- [ ] Verify total sales amount

### Channel Impact Tab
- [ ] Verify ROI chart displays correctly
- [ ] Check for appropriate error bars/confidence intervals
- [ ] Verify spend efficiency visualization

### Media Mix Curves Tab (New Feature)
- [ ] Verify tab is present and selectable
- [ ] Check that response curves visualization loads
- [ ] Verify current spend positions are marked on the curves
- [ ] Check that marginal returns chart displays
- [ ] Verify channel parameter summary cards display:
  - [ ] Max Effect (L) values
  - [ ] Growth Rate (k) values
  - [ ] Half-Saturation Point (x0) values
  - [ ] Saturation percentage indicators

### Budget Optimization Tab
- [ ] Click "Optimize Budget" button
- [ ] Enter test budget amount
- [ ] Click "Run Optimization" button
- [ ] Verify optimization results appear:
  - [ ] Channel allocation recommendations
  - [ ] Expected outcome
  - [ ] Lift percentage
- [ ] Check comparison with current allocation

**Network/Console Checks:**
- [ ] Verify POST request to `/api/models/{id}/optimize-budget`
- [ ] Check server logs for:
  - [ ] `mmm_optimizer_service.py is being called`
  - [ ] `Using saturation parameters: L, k, x0`
  - [ ] `Enhanced optimizer with proper scaling`

## 7. Variant Testing

- [ ] Create another model with different saturation parameters
- [ ] Train the model
- [ ] Compare results with the first model
- [ ] Verify parameters impact the Media Mix Curves visualization
- [ ] Run budget optimization on both models and compare results

## Common Issues and Solutions

### Data Format Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Incorrect date format | Date parsing errors in console | Ensure CSV has DD/MM/YYYY format; check dataSources.ts for `dayfirst=True` parameter |
| Comma-separated numbers | Channel values parsed as strings/NaN | Verify number cleaning in dataSources.ts, check for `String(ex).replace(/,/g, '')` |
| Insufficient data rows | Training error or poor results | Ensure at least 8 weeks of continuous data |

### Model Training Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| TensorVariable dims error | "AttributeError: 'TensorVariable' object has no attribute 'dims'" | Verify global patch is applied correctly in fit_mmm_fixed_params.py |
| Saturation parameter issues | Extreme values in Media Mix Curves | Check default values in model-setup.tsx, ensure reasonable ranges |
| Model training timeout | Training stuck at a percentage | Check Python script execution, may need to increase timeout or optimize code |

### Results Visualization Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing analytics data | Blank charts/sections | Check transformMMMResults function in modelTraining.ts |
| Media Mix Curves not showing | "No saturation curve data available" | Verify model.config.channels contains saturation parameters |
| Budget optimizer not working | Optimization error or unrealistic results | Check payload format and verify mmm_optimizer_service.py execution |

## Export and Reporting Tests

- [ ] Test PDF export functionality (if available)
- [ ] Test sharing/exporting of recommendations
- [ ] Verify any email reporting features

---

## Sign-off Criteria

All test results must be documented with screenshots and logs. The workflow is considered validated when:

1. All functional steps complete successfully
2. Results show realistic and consistent values
3. Media Mix Curves display correctly with expected patterns
4. Budget optimization provides reasonable recommendations
5. No critical errors appear in console/server logs

**Tester Name:** _________________________

**Test Date:** _________________________

**Sign-off:** _________________________