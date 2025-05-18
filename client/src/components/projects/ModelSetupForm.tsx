import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { AlertCircle, CheckIcon, X } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import api from "@/lib/api";

interface ModelSetupFormProps {
  projectId: number;
  currentStep: number;
  onStepChange: (step: number) => void;
  onComplete: (modelData: any) => void;
}

export default function ModelSetupForm({ 
  projectId, 
  currentStep, 
  onStepChange,
  onComplete
}: ModelSetupFormProps) {
  const queryClient = useQueryClient();
  const [formError, setFormError] = useState<string | null>(null);

  // Fetch data sources for this project
  const { data: dataSources } = useQuery({
    queryKey: [`/api/projects/${projectId}/data-sources`],
  });

  // Extract columns from first data source for mapping
  const availableColumns: string[] = dataSources?.[0]?.metricColumns || [];

  // Form validation schema for each step
  const businessQuestionsSchema = z.object({
    name: z.string().min(3, { message: "Model name is required" }),
    businessObjective: z.enum(["sales_growth", "brand_awareness", "conversion_rate", "cost_efficiency"]),
    channelFocus: z.array(z.string()).min(1, { message: "Select at least one channel" }),
    timeHorizon: z.enum(["short_term", "medium_term", "long_term"]),
  });

  const dataMappingSchema = z.object({
    dateColumn: z.string().min(1, { message: "Date column is required" }),
    salesColumn: z.string().min(1, { message: "Sales column is required" }),
    channelColumns: z.record(z.string().min(1)).refine(val => Object.keys(val).length > 0, {
      message: "At least one channel must be mapped"
    }),
    controlColumns: z.record(z.string().optional()),
  });

  const modelParametersSchema = z.object({
    adstockDecay: z.number().min(0.01).max(0.99),
    saturationRate: z.number().min(0.01).max(0.99),
    regularization: z.enum(["none", "low", "medium", "high"]),
    seasonalityDetection: z.boolean(),
    crossChannelEffects: z.boolean(),
  });

  const reviewSchema = z.object({
    acceptTerms: z.boolean().refine(val => val === true, {
      message: "You must accept the terms to continue"
    }),
  });

  // Combined schema based on current step
  const getSchemaForStep = (step: number) => {
    switch (step) {
      case 0: return businessQuestionsSchema;
      case 1: return dataMappingSchema;
      case 2: return modelParametersSchema;
      case 3: return reviewSchema;
      default: return businessQuestionsSchema;
    }
  };

  // Form setup
  const form = useForm({
    resolver: zodResolver(getSchemaForStep(currentStep)),
    defaultValues: {
      // Business Questions
      name: "",
      businessObjective: "sales_growth" as const,
      channelFocus: ["social", "search", "display"],
      timeHorizon: "medium_term" as const,
      
      // Data Mapping
      dateColumn: "",
      salesColumn: "",
      channelColumns: {},
      controlColumns: {},
      
      // Model Parameters
      adstockDecay: 0.5,
      saturationRate: 0.7,
      regularization: "medium" as const,
      seasonalityDetection: true,
      crossChannelEffects: false,
      
      // Review
      acceptTerms: false,
    }
  });

  // Create model mutation
  const createModel = useMutation({
    mutationFn: (modelData: any) => {
      return api.createModel({
        projectId,
        name: modelData.name,
        adstockSettings: {
          decay: modelData.adstockDecay,
        },
        saturationSettings: {
          rate: modelData.saturationRate,
        },
        controlVariables: modelData.controlColumns,
        responseVariables: {
          salesColumn: modelData.salesColumn,
          dateColumn: modelData.dateColumn,
          channelColumns: modelData.channelColumns,
        }
      });
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: [`/api/projects/${projectId}/models`] });
      onComplete(response);
    },
    onError: (error: any) => {
      setFormError(error.message || "Failed to create model. Please try again.");
    }
  });

  // Get fields for current step
  const renderFieldsForStep = () => {
    switch (currentStep) {
      case 0: // Business Questions
        return (
          <>
            <FormField
              control={form.control}
              name="name"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Model Name</FormLabel>
                  <FormControl>
                    <Input placeholder="Q2 Marketing Analysis" {...field} />
                  </FormControl>
                  <FormDescription>
                    Give your model a descriptive name.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="businessObjective"
              render={({ field }) => (
                <FormItem className="space-y-3">
                  <FormLabel>What's your primary business objective?</FormLabel>
                  <FormControl>
                    <RadioGroup
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                      className="space-y-1"
                    >
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="sales_growth" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Maximize Sales Growth
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="brand_awareness" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Increase Brand Awareness
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="conversion_rate" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Improve Conversion Rate
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="cost_efficiency" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Optimize Cost Efficiency
                        </FormLabel>
                      </FormItem>
                    </RadioGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="channelFocus"
              render={() => (
                <FormItem>
                  <div className="mb-4">
                    <FormLabel className="text-base">Which channels do you want to analyze?</FormLabel>
                    <FormDescription>
                      Select all the marketing channels you want to include in your analysis.
                    </FormDescription>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    {[
                      { id: "social", label: "Social Media" },
                      { id: "search", label: "Search Ads" },
                      { id: "display", label: "Display Ads" },
                      { id: "email", label: "Email Marketing" },
                      { id: "tv", label: "TV Advertising" },
                      { id: "print", label: "Print Media" },
                      { id: "radio", label: "Radio" },
                      { id: "ooh", label: "Out of Home" }
                    ].map((channel) => (
                      <FormField
                        key={channel.id}
                        control={form.control}
                        name="channelFocus"
                        render={({ field }) => {
                          return (
                            <FormItem
                              key={channel.id}
                              className="flex flex-row items-start space-x-3 space-y-0"
                            >
                              <FormControl>
                                <Checkbox
                                  checked={field.value?.includes(channel.id)}
                                  onCheckedChange={(checked) => {
                                    return checked
                                      ? field.onChange([...field.value, channel.id])
                                      : field.onChange(
                                          field.value?.filter(
                                            (value) => value !== channel.id
                                          )
                                        )
                                  }}
                                />
                              </FormControl>
                              <FormLabel className="font-normal">
                                {channel.label}
                              </FormLabel>
                            </FormItem>
                          )
                        }}
                      />
                    ))}
                  </div>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="timeHorizon"
              render={({ field }) => (
                <FormItem className="space-y-3">
                  <FormLabel>What timeframe are you interested in?</FormLabel>
                  <FormControl>
                    <RadioGroup
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                      className="space-y-1"
                    >
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="short_term" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Short-term effects (days to weeks)
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="medium_term" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Medium-term effects (weeks to months)
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="long_term" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Long-term effects (months to years)
                        </FormLabel>
                      </FormItem>
                    </RadioGroup>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </>
        );

      case 1: // Data Mapping
        return (
          <>
            <Alert className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Select the appropriate columns from your data that correspond to dates, sales metrics, and marketing channels.
              </AlertDescription>
            </Alert>

            <FormField
              control={form.control}
              name="dateColumn"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Date Column</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select date column" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {availableColumns
                        .filter(col => col.toLowerCase().includes('date') || col.toLowerCase().includes('time') || col.toLowerCase().includes('day'))
                        .map(column => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    The column that contains your dates (e.g., "Date", "Week", "Month")
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="salesColumn"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Sales/Conversion Column</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select sales column" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {availableColumns
                        .filter(col => !col.toLowerCase().includes('date') && !col.toLowerCase().includes('time'))
                        .map(column => (
                          <SelectItem key={column} value={column}>
                            {column}
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                  <FormDescription>
                    The column that contains your target metric (e.g., "Sales", "Revenue", "Conversions")
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium mb-1">Map Marketing Channels</h3>
                <p className="text-sm text-muted-foreground">
                  For each marketing channel, select the corresponding column from your data.
                </p>
              </div>

              {form.getValues("channelFocus").map((channel) => {
                const channelLabel = {
                  social: "Social Media",
                  search: "Search Ads",
                  display: "Display Ads",
                  email: "Email Marketing",
                  tv: "TV Advertising",
                  print: "Print Media",
                  radio: "Radio",
                  ooh: "Out of Home"
                }[channel] || channel;

                return (
                  <FormField
                    key={channel}
                    control={form.control}
                    name={`channelColumns.${channel}`}
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>{channelLabel} Column</FormLabel>
                        <Select onValueChange={field.onChange} defaultValue={field.value}>
                          <FormControl>
                            <SelectTrigger>
                              <SelectValue placeholder={`Select column for ${channelLabel}`} />
                            </SelectTrigger>
                          </FormControl>
                          <SelectContent>
                            {availableColumns
                              .filter(col => !col.toLowerCase().includes('date') && !col.toLowerCase().includes('time'))
                              .map(column => (
                                <SelectItem key={column} value={column}>
                                  {column}
                                </SelectItem>
                              ))}
                          </SelectContent>
                        </Select>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                );
              })}

              <Separator className="my-6" />

              <div>
                <h3 className="text-sm font-medium mb-1">Control Variables (Optional)</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Map additional factors that may influence your results, such as promotions, seasonality, etc.
                </p>

                <div className="space-y-4">
                  {["Promotions", "Holidays", "Competitor Activity"].map((control) => {
                    const controlId = control.toLowerCase().replace(" ", "_");
                    return (
                      <FormField
                        key={controlId}
                        control={form.control}
                        name={`controlColumns.${controlId}`}
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>{control} Column (Optional)</FormLabel>
                            <Select onValueChange={field.onChange} defaultValue={field.value}>
                              <FormControl>
                                <SelectTrigger>
                                  <SelectValue placeholder={`Select column for ${control} (optional)`} />
                                </SelectTrigger>
                              </FormControl>
                              <SelectContent>
                                <SelectItem value="">None</SelectItem>
                                {availableColumns
                                  .filter(col => !col.toLowerCase().includes('date') && !col.toLowerCase().includes('time'))
                                  .map(column => (
                                    <SelectItem key={column} value={column}>
                                      {column}
                                    </SelectItem>
                                  ))}
                              </SelectContent>
                            </Select>
                            <FormDescription>
                              {control === "Promotions" 
                                ? "Column indicating when promotions were active (0/1 or TRUE/FALSE)" 
                                : control === "Holidays" 
                                ? "Column indicating holiday periods" 
                                : "Column with competitor activity data"
                              }
                            </FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    );
                  })}
                </div>
              </div>
            </div>
          </>
        );

      case 2: // Model Parameters
        return (
          <>
            <Alert className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Adjust these advanced parameters to fine-tune your model. The default values are recommended for most cases.
              </AlertDescription>
            </Alert>

            <FormField
              control={form.control}
              name="adstockDecay"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Adstock Decay Rate: {field.value.toFixed(2)}</FormLabel>
                  <FormControl>
                    <Slider
                      min={0.01}
                      max={0.99}
                      step={0.01}
                      defaultValue={[field.value]}
                      onValueChange={(values) => field.onChange(values[0])}
                    />
                  </FormControl>
                  <FormDescription>
                    Controls how quickly marketing effects decay over time. Lower values mean longer-lasting effects.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="saturationRate"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Saturation Rate: {field.value.toFixed(2)}</FormLabel>
                  <FormControl>
                    <Slider
                      min={0.01}
                      max={0.99}
                      step={0.01}
                      defaultValue={[field.value]}
                      onValueChange={(values) => field.onChange(values[0])}
                    />
                  </FormControl>
                  <FormDescription>
                    Controls diminishing returns on marketing spend. Higher values mean faster saturation (diminishing returns).
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <FormField
              control={form.control}
              name="regularization"
              render={({ field }) => (
                <FormItem className="space-y-3">
                  <FormLabel>Regularization Strength</FormLabel>
                  <FormControl>
                    <RadioGroup
                      onValueChange={field.onChange}
                      defaultValue={field.value}
                      className="space-y-1"
                    >
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="none" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          None - No regularization
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="low" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Low - Slight prevention of overfitting
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="medium" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          Medium - Balanced approach (recommended)
                        </FormLabel>
                      </FormItem>
                      <FormItem className="flex items-center space-x-3 space-y-0">
                        <FormControl>
                          <RadioGroupItem value="high" />
                        </FormControl>
                        <FormLabel className="font-normal">
                          High - Strong prevention of overfitting
                        </FormLabel>
                      </FormItem>
                    </RadioGroup>
                  </FormControl>
                  <FormDescription>
                    Controls how strongly the model prevents overfitting to the training data.
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />

            <div className="space-y-4">
              <FormField
                control={form.control}
                name="seasonalityDetection"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-start space-x-3 space-y-0 rounded-md border p-4">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>
                        Enable Seasonality Detection
                      </FormLabel>
                      <FormDescription>
                        Automatically detect and account for seasonal patterns in your data.
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="crossChannelEffects"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-start space-x-3 space-y-0 rounded-md border p-4">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>
                        Analyze Cross-Channel Effects
                      </FormLabel>
                      <FormDescription>
                        Model interactions between different marketing channels. May increase model complexity.
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />
            </div>
          </>
        );

      case 3: // Review & Run
        return (
          <>
            <Alert className="mb-6">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Review your model configuration before starting the training process. Model training may take several minutes to complete.
              </AlertDescription>
            </Alert>

            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Business Objectives</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="font-medium">Model Name:</dt>
                      <dd>{form.getValues("name")}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Primary Objective:</dt>
                      <dd>{form.getValues("businessObjective").replace("_", " ")}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Time Horizon:</dt>
                      <dd>{form.getValues("timeHorizon").replace("_", " ")}</dd>
                    </div>
                    <div>
                      <dt className="font-medium mb-1">Selected Channels:</dt>
                      <dd className="flex flex-wrap gap-1">
                        {form.getValues("channelFocus").map(channel => (
                          <span key={channel} className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-slate-100">
                            {channel.replace("_", " ")}
                          </span>
                        ))}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Data Mapping</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="font-medium">Date Column:</dt>
                      <dd>{form.getValues("dateColumn")}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Sales Column:</dt>
                      <dd>{form.getValues("salesColumn")}</dd>
                    </div>
                    <div>
                      <dt className="font-medium mb-1">Channel Columns:</dt>
                      <dd className="space-y-1">
                        {Object.entries(form.getValues("channelColumns")).map(([channel, column]) => (
                          <div key={channel} className="flex justify-between text-sm">
                            <span>{channel.replace("_", " ")}:</span>
                            <span>{column || "Not mapped"}</span>
                          </div>
                        ))}
                      </dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Advanced Parameters</CardTitle>
                </CardHeader>
                <CardContent>
                  <dl className="space-y-2">
                    <div className="flex justify-between">
                      <dt className="font-medium">Adstock Decay Rate:</dt>
                      <dd>{form.getValues("adstockDecay").toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Saturation Rate:</dt>
                      <dd>{form.getValues("saturationRate").toFixed(2)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Regularization:</dt>
                      <dd>{form.getValues("regularization")}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Seasonality Detection:</dt>
                      <dd>{form.getValues("seasonalityDetection") ? "Enabled" : "Disabled"}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-medium">Cross-Channel Effects:</dt>
                      <dd>{form.getValues("crossChannelEffects") ? "Enabled" : "Disabled"}</dd>
                    </div>
                  </dl>
                </CardContent>
              </Card>

              <FormField
                control={form.control}
                name="acceptTerms"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-start space-x-3 space-y-0 rounded-md border p-4">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel>
                        I understand that model training may take several minutes
                      </FormLabel>
                      <FormDescription>
                        You will be notified when the model training is complete. You can work on other projects while waiting.
                      </FormDescription>
                    </div>
                  </FormItem>
                )}
              />
            </div>
          </>
        );

      default:
        return null;
    }
  };

  // Handle next step
  const handleNext = async () => {
    try {
      // Validate current step
      await form.trigger();
      if (!form.formState.isValid) return;

      // If last step, handle submission
      if (currentStep === 3) {
        const modelData = form.getValues();
        createModel.mutate(modelData);
        return;
      }

      // Move to next step
      onStepChange(currentStep + 1);
    } catch (error) {
      console.error("Form error:", error);
    }
  };

  // Handle previous step
  const handlePrevious = () => {
    if (currentStep > 0) {
      onStepChange(currentStep - 1);
    }
  };

  return (
    <div className="space-y-6">
      {formError && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{formError}</AlertDescription>
        </Alert>
      )}

      <Form {...form}>
        <form className="space-y-6">
          {renderFieldsForStep()}

          <div className="flex justify-between pt-4">
            <Button 
              type="button" 
              variant="outline" 
              onClick={handlePrevious}
              disabled={currentStep === 0}
            >
              Previous
            </Button>
            <Button 
              type="button" 
              onClick={handleNext}
              disabled={createModel.isPending}
            >
              {currentStep === 3 
                ? (createModel.isPending ? "Starting Training..." : "Start Model Training") 
                : "Next"}
            </Button>
          </div>
        </form>
      </Form>
    </div>
  );
}
