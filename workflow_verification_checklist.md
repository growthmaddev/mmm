# MMM Workflow Verification Checklist

## 1. Project Setup & Data Upload
- [ ] Create new project
- [ ] Upload `dankztestdata_v2.csv`
- [ ] Verify date parsing (DD/MM/YYYY format)
- [ ] Check numeric columns cleaned (commas removed)

## 2. Column Mapping
- [ ] Date column: `Date`
- [ ] Target/Metric column: `Sales`
- [ ] Channel columns mapped:
  - [ ] PPCBrand_Spend
  - [ ] PPCNonBrand_Spend
  - [ ] PPCShopping_Spend
  - [ ] FBReach_Spend
  - [ ] OfflineMedia_Spend
- [ ] Control variable: `interestrate_control`

## 3. Model Configuration
- [ ] Set model name
- [ ] Configure adstock (alpha values)
- [ ] Configure saturation (L, k, x0 values)
- [ ] Enable control variables

## 4. Model Training
- [ ] Click "Create Model"
- [ ] Verify uses `fit_mmm_ridge.py` (check server logs)
- [ ] Monitor progress updates
- [ ] Training completes successfully

## 5. Results Page - Overview Tab
- [ ] Model accuracy shows real R-squared (~7.7%)
- [ ] Top channel identified correctly (PPCBrand)
- [ ] ROI values displayed (PPCBrand ~73x)
- [ ] Recommendations generated

## 6. Results Page - Sales Decomposition Tab
- [ ] Pie chart displays with correct percentages
- [ ] Base sales ~77%, Marketing ~23%
- [ ] Channel names display correctly
- [ ] Total sales amount shown

## 7. Results Page - Channel Impact Tab
- [ ] ROI bar chart displays all channels
- [ ] PPCBrand shows highest ROI (~73x)
- [ ] Channel performance table populated
- [ ] Efficiency matrix (quadrant chart) works

## 8. Results Page - Media Mix Curves Tab
- [ ] Saturation curves display for each channel
- [ ] Current spend position marked
- [ ] Parameters (L, k, x0) shown
- [ ] NOT showing "No saturation curve data available"

## 9. Results Page - Budget Optimization Tab
- [ ] Links to budget optimizer
- [ ] Passes model ID correctly

## Test Data Locations
- Data file: `uploads/dankztestdata_v2.csv`
- Config: Will be created during model setup
- Expected R-squared: ~7.7%
- Expected top ROI: PPCBrand ~73x

## Server Logs to Monitor
Look for:
- "Using Ridge regression MMM"
- "Config being passed to UI"
- No errors about missing dims or PyMC compatibility

## Browser Console Errors to Watch For
- Missing data for curve visualization
- Undefined saturation parameters
- Config structure issues

## Integration Issues to Document
- [ ] Verify the transformer preserves Ridge regression data correctly
- [ ] Check budget optimizer respects channel characteristics
- [ ] Ensure saturation curves receive config parameters (L, k, x0)