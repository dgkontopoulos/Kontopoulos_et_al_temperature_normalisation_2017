#!/usr/bin/Rscript

# This script first generates 1,000 simulated TPCs without the effects of 
# Metabolic Cold Adaptation.
#
# It later shows that even on this population of curves, the overestimation of 
# B0 can mimic the effects of MCA, while B(T_ref) does not.
#
# Usage: ./run_simulation.R

library(ggalt)
library(ggplot2)
library(minpack.lm)

#############################
# F  U  N  C  T  I  O  N  S #
#############################

# This function calculates the hypothetical B0 value of a simulated curve, by 
# returning the height of the curve at the data point closest to T_ref = 7°C.
calc_B0 <- function(distr)
{
    distr$y[abs(distr$x-7)==min(abs(distr$x-7))]
}

# This function calculates the hypothetical T_pk value of a simulated curve, by 
# returning the temperature at which the curve peaks.
calc_T_pk <- function(distr)
{
    distr$x[distr$y == max(distr$y)]
}

# This function generates simulated TPCs, in which no trace of metabolic cold 
# adaptation is present.
simulate_data <- function()
{

    # Set the random seed to 1337.
    set.seed(1337)

    # Initialise vectors to hold required information for each TPC.
    B0 <- c()
    T_pk <- c()
    
    IDs <- c()
    TraitVal <- c()
    TempVal <- c()
	
    counter <- 0
    
    # While the total number of simulated curves is less than 1000 ...
    while ( length(B0) < 1000 )
    {
    
	# Generate the parameters of a beta distribution by sampling from 
	# normal distributions.
	i <- rnorm(1,10, 3)
	k <- rnorm(1,4,2)
	m <- rnorm(1, 25, 4)
	n <- rnorm(1, 3, 0.8)

	# Try to sample from the beta distribution.
	b_dist <- NA
	try(b_dist <- density(rbeta(100000, i, i - k) * m))
	
	# If that was successful and the curve is negatively skewed ...
	if (!(is.na(b_dist)) && i > (i - k))
	{
	
	    # Increment the TPC counter.
	    counter <- counter + 1
	    
	    # Increase the height of the curve by a random factor n.
	    b_dist$y <- b_dist$y + n
	
	    # Calculate the B0 and T_pk values for this TPC.
	    B0 <- c(B0, calc_B0(b_dist))
	    T_pk <- c(T_pk, calc_T_pk(b_dist))
	    
	    # Save the ID, trait values, and temperature values for this TPC.
	    IDs <- c(IDs, rep(counter, length(b_dist$x)))
	    TraitVal <- c(TraitVal, b_dist$y)
	    TempVal <- c(TempVal, b_dist$x + 273.15)
	}
    }

    # Print the lowest T_pk value of these TPCs.
    print(min(T_pk))

    # Calculate the correlation between B0 and T_pk in this ensemble of TPCs.
    print(cor.test(B0,T_pk))

    # Return the results of the simulation.
    dataset <- data.frame(IDs = IDs, TraitVal = TraitVal, TempVal = TempVal)	
    return(dataset)
}

# This function generates starting values for the various parameters of the 
# Sharpe-Schoolfield model.
gen.starting.values <- function(tmpData)
{

    # Set the starting value of E_D to 2 eV.
    E_D_strt <- 2
    
    # Find the highest trait value for this particular dataset.
    Trait_max <- mean(max(tmpData[, "TraitVal"]), na.rm = TRUE)

    # Set the starting value of T_pk to the temperature where the highest 
    # trait value is observed.
    T_pk_strt <- max(
	tmpData[
	    which(
		tmpData[, "TraitVal"] == mean(
		    max(tmpData[, "TraitVal"]), na.rm = TRUE
		)
	    ), 
	    "TempVal"
	], na.rm=TRUE
    )

    # Extract the subset of the data that correspond to the rising part of 
    # the curve.
    tmptmpData <- subset(tmpData, tmpData[, "TempVal"] < T_pk_strt)

    # Get the minimum trait value at the rise of the curve as the starting 
    # value of B0.
    B0_strt <- min(
	tmptmpData[
	    which(
		tmptmpData[, "TempVal"] == mean(
		    min(tmptmpData[, "TempVal"])
		)
	    ), "TraitVal"
	], na.rm=TRUE
    )

    # The starting value of E is the coefficient of the linear regression of 
    # Trait values up to peak ~ 1/(k * Temperature).
    x <- 1/(k * tmptmpData[, "TempVal"])
    y <- log(tmptmpData[, "TraitVal"])
    fit <- lm(y ~ x)
    E_strt <- abs(fit$coefficients[2])

    return(list(E_D_strt = E_D_strt, T_pk_strt = T_pk_strt, B0_strt = B0_strt, 
        E_strt = E_strt))
}

# This function fits the Sharpe-Schoolfield model to data, using a T_ref of 
# 7°C.
schoolf <- function(B0, E, E_D, T_pk, temp)
{
    # Parameters
    # B0     : Trait performance at low temperature
    # E      : Activation energy 
    # E_D    : De-Activation energy 
    # T_pk   : T at which enzyme is 50% active and 50% high-temperature suppressed
    
    return(
	log(B0 * exp(-E * ((1/(k*temp)) - (1/(k*280.15))))/(
		1 + (E/(E_D - E)) * exp(E_D/k * (1/T_pk - 1/temp))
	    )
	)
    )
}

############################
# M  A  I  N    C  O  D  E #
############################

options(warn=-1)

# Simulate TPCs.
dataset <- simulate_data()

options(warn=0)

# Declare the Boltzmann's constant as a global variable.
assign("k", 8.617 * 10^-5, envir = .GlobalEnv)

# Set the conditions for NLLS fitting.
cont <- nls.control(maxiter = 100000, tol = 1e-12, minFactor = 1/1024, 
    printEval = FALSE, warnOnly = TRUE)

# Initialise vectors to store the final parameter estimates.
B0 <- c()
B_T_ref <- c()
T_pk <- c()

# Iterate over the 1000 simulated TPCs.
for ( j in 1:1000 )
{
    print(j)

    # Get the subset of the data that correspond to this particular TPC.
    current_dataset <- dataset[dataset$IDs == j,]
    
    # Obtain starting values for the parameters of the Sharpe-Schoolfield model.
    starting.values <- gen.starting.values(current_dataset)
    B0_strt <- starting.values$B0_strt
    E_strt <- starting.values$E_strt
    E_D_strt <- starting.values$E_D_strt
    T_pk_strt <- starting.values$T_pk_strt
    
    # Try to fit the model using the Levenberg-Marquardt algorithm.
    Schoolfit <- NA
    
    try(
	Schoolfit <- nlsLM(log(TraitVal) ~ schoolf(B0, E, E_D, T_pk, TempVal), 
	current_dataset, 
	start = list(B0 = B0_strt, E = E_strt, E_D = E_D_strt, T_pk = T_pk_strt),
        lower = c(0, 0, 0, 273.15),
        upper = c(Inf, 30, 50, 273.15 + 150), 
        control = cont)
    )
        
    # If fitting was successful ...
    if ( !is.na(Schoolfit) )
    {
    
	# Extract the B0 estimate and manually calculate B(T_ref).
	B0 <- c(B0, coef(Schoolfit)["B0"])
	B_T_ref <- c(
	    B_T_ref,
	    coef(Schoolfit)["B0"] / (
		1 + ( 
		    coef(Schoolfit)["E.x"] / (
			coef(Schoolfit)["E_D"] - coef(Schoolfit)["E.x"]
		    )
		) *
		exp(
		    (
			coef(Schoolfit)["E_D"] / k
		    ) * 
		    (
			1 / coef(Schoolfit)["T_pk"] - 1 / 280.15
		    )
		)
	    )
	)
	
	# Extract the T_pk estimate.
	T_pk <- c(T_pk, coef(Schoolfit)["T_pk"])
    }
}

# Collect the results and separate hypothetical species between those that 
# have T_pk values below and above 15°C.
B0_df <- data.frame(y = B0, T_pk = T_pk, Var = rep('B0', length(B0)))
B_T_ref_df <- data.frame(y = B_T_ref, T_pk = T_pk, 
    Var = rep('B(T_ref)', length(B_T_ref)))

results_df <- rbind(B0_df, B_T_ref_df)
results_df$Group[results_df$T_pk < (273.15 + 15)] <- "[8.23, 15)"
results_df$Group[results_df$T_pk >= (273.15 + 15)] <- "[15, 36)"
results_df$Group <- factor(results_df$Group, levels = c("[8.23, 15)", "[15, 36)"), 
    ordered = TRUE)

# Perform two-sample Kolmogorov-Smirnov tests between i) B0 for species with 
# T_pk below and above 15°C, and ii) B(T_ref) for species with T_pk below and 
# above 15°C.
ks_B0 <- ks.test(
    x = results_df$y[results_df$Group == "[8.23, 15)" & results_df$Var == "B0"],
    y = results_df$y[results_df$Group == "[15, 36)"& results_df$Var == "B0"]
)

ks_B_T_ref <- ks.test(
    x = results_df$y[
	results_df$Group == "[8.23, 15)" & results_df$Var == "B(T_ref)"],
    y = results_df$y[
	results_df$Group == "[15, 36)"& results_df$Var == "B(T_ref)"]
)

# Generate violin plots of the results using ggplot2 (panel A).
pdf(file = "../Results/simulation.pdf", width = 3, height = 2.25, 
    colormodel = 'cmyk')

ggplot(results_df, aes(x = Group, y = y)) + geom_violin(aes(fill = Var), 
    draw_quantiles = c(0.5), width = 0.75, position=position_dodge(1)) +
    ylab(expression("Performance at 7°C (" * s^-1 * ")")) +
    xlab(expression(italic(T)[pk] ~ "(°C)")) +
    theme_bw() +
    scale_fill_manual(labels = expression(italic(B)[0], 
	italic(B) * '(' * italic(T)[ref] * ')'), 
	values = c("#a6cee3", "#5fd35f")) +
    theme(
	plot.margin = unit(c(0,0,0,0), "cm"),
	axis.text.y = element_text(size = 9, angle=90, hjust = 1),
	axis.text.x = element_text(size = 9),
	plot.title = element_text(size=12, margin=margin(b = 2, unit = "pt"), 
	    face = 'bold'),
	axis.title.x = element_text(size = 10),
	axis.title.y = element_text(size = 10),
	panel.grid.major = element_blank(),
	panel.grid.minor = element_blank(),
	legend.title = element_blank(),
	legend.key.size = unit(0.75, 'lines'),
	legend.position=c(.85,.86),
	legend.margin=unit(0,"cm")
    ) + 
    annotate("text", x = 1.362, y = 6.85, 
	label = "italic(D)==0.18 *'; '~italic(p)==1.7*'·'*10^-6", size = 3, 
	color = "#a6cee3", parse = TRUE) + 
    annotate("text", x = 1.27, y = 6.15, 
	label = "italic(D)==0.07 *'; '~italic(p)==0.21", size = 3, 
	color = "#5fd35f", parse = TRUE)

dev.off()

# Perform correlations between i) B0 and T_pk, and ii) B(T_ref) and T_pk.
print(cor.test(B0, T_pk))
print(cor.test(B_T_ref, T_pk))

results_df_2 <- rbind(B0_df, B_T_ref_df)
results_df_2$Var <- as.character(results_df_2$Var)
results_df_2$Var[results_df_2$Var == "B0"] <- 'italic(B)[0]'
results_df_2$Var[results_df_2$Var == "B(T_ref)"] <- 'italic(B)*"("*italic(T)[ref]*")"'

# Generate correlation plots of the results using ggplot2 (panel B).
pdf(file = "../Results/simulation2.pdf", width = 3, height = 3, 
    colormodel = 'cmyk')
ggplot(results_df_2, aes(x = T_pk - 273.15, y = y)) +
    geom_point(size=0.6, show.legend = FALSE) +
    stat_bkde2d(aes(fill=..level..), alpha = 0.7, geom="polygon", show.legend = FALSE) +
    scale_fill_gradientn(colours = terrain.colors(100)) +
    facet_wrap(~Var, ncol = 1, nrow = 2, scales = "free_x", labeller=label_parsed) +
    ylab(expression("Performance at 7°C (" * s^-1 * ")")) +
    xlab(expression(italic(T)[pk] ~ "(°C)")) +
    theme_bw() +
    theme(
	plot.margin = unit(c(0,0,0,0), "cm"),
	axis.text.y = element_text(size = 9, angle=90, hjust = 1),
	axis.text.x = element_text(size = 9),
	plot.title = element_text(size=12, margin=margin(b = 2, unit = "pt"), 
	    face = 'bold'),
	axis.title.x = element_text(size = 10),
	axis.title.y = element_text(size = 10),
	panel.grid.major = element_blank(),
	panel.grid.minor = element_blank()
    )
dev.off()
