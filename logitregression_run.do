*=======================================================
*
*        Run logistic regression
*
*=======================================================
* 

  global path "/Users/HaoLI/Stata/credit"    
  global D    "$path/data"      //范例数据
  global R    "$path/refs"      //参考文献
  global Out  "$path/out"       //结果：图形和表格
  adopath +   "$PP/adofiles"    //外部命令 
  cd "$D"
  
 
  import delimited data1020use.csv, clear // data before this dofile
  //destring ttlamt_first8mon	ttlcnt_first8mon , replace force
  //use "data1020.dta", clear  // data processed after this dofile
 
 
logit default_geq_1 gender age edu_level_1 edu_level_2 edu_level_3 edu_level_5 edu_level_6 edu_level_7 housing_flag log_salary_level midnight_amt_mean  night_amt_mean  morning_amt_mean 
est store Baseline1

logit default_geq_1 debit_amt_mean credit_amt_mean dcwt_amtpcnt95p dcwt_ordpmon99p dcmwth  cmwth_mcht_cnt ltry_ttlamt_ttlexpd realestate_mon_amtpexpd  dbldg vdsh_effmonratio0 water_effmonratio0
est store Time1

logit default_geq_1  gender age edu_level_1 edu_level_2 edu_level_3 edu_level_5 edu_level_6 edu_level_7  housing_flag log_salary_level midnight_amt_mean  night_amt_mean  morning_amt_mean debit_amt_mean credit_amt_mean dcwt_amtpcnt95p dcwt_ordpmon99p dcmwth  cmwth_mcht_cnt ltry_ttlamt_ttlexpd realestate_mon_amtpexpd  dbldg vdsh_effmonratio0 water_effmonratio0
est store Base_Time1

logit default_geq_1  gender age edu_level_1 edu_level_2 edu_level_3 edu_level_5 edu_level_6 edu_level_7  housing_flag log_salary_level midnight_amt_mean  night_amt_mean  morning_amt_mean debit_amt_mean credit_amt_mean dcwt_amtpcnt95p dcwt_ordpmon99p dcmwth  cmwth_mcht_cnt ltry_ttlamt_ttlexpd realestate_mon_amtpexpd  dbldg vdsh_effmonratio0 water_effmonratio0 i.city_index
est store Base_Time1_FE

reg2docx Baseline1 Time1 Base_Time1 Base_Time1_FE using Tables_1210.docx, drop(*.city_index) ///
	 scalars(N r2_p(%9.4f)) b(%9.3f) t(%7.2f) ///
      title("Logit Regression Result") ///
	  mtitles("Baseline" "Consumption" "Full Model" "Full Model, further controls") replace 	
 
 
 
 outreg2 using result1020.txt
  
  
   
  // demographic+三个时间amt mean 作为base
  
  
  
  morning_amt_vol  midnight_amt_vol night_amt_vol //作为alternative, 可以替代三个amt_mean
  
  
  midnight_freq_mean morning_freq_mean night_freq_mean
  
   midnight_amt_vol morning_amt_vol afternoon_amt_vol night_amt_vol
  midnight_freq_vol morning_freq_vol afternoon_freq_vol night_freq_vol
  
