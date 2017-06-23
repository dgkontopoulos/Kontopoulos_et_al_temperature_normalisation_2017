#!/usr/bin/env perl

# This script builds the conditional inference tree model from the TPC data
# and evaluates its performance against a testing dataset.
#
# Usage: ./build_ctree_model.pl

use strict;
use warnings;

use feature qw(say);

use Readonly;
use Statistics::R;
use Text::CSV_XS qw(csv);

# Declare Boltzmann's constant and T_ref as fixed values.
Readonly::Scalar my $K     => 8.617 * 10**(-5);
Readonly::Scalar my $T_REF => 273.15;

# Read the training dataset of TPC fits.
my $training_dataset = csv(
    in       => '../Results/fits_Sharpe_Schoolfield.csv',
    headers  => 'skip',
    sep_char => "\t",
    fragment => 'col=1-2;4-7'
);

# Read the testing dataset of TPC fits.
my $testing_dataset = csv(
    in       => '../Results/fits_Sharpe_Schoolfield_test.csv',
    headers  => 'skip',
    sep_char => "\t",
    fragment => 'col=1-2;4-7'
);

# Initialise counters to keep track of the number of processed fits.
my $counter_training = 0;
my $counter_testing  = 0;

# Initialise arrays for variables of the training and testing dataset.
my (
    @id,                    @t_pk_minus_t_ref,     @t_h_minus_t_ref,
    @t_pk_minus_t_h,        @outcome,              @id_test,
    @t_pk_minus_t_ref_test, @t_h_minus_t_ref_test, @t_pk_minus_t_h_test,
    @outcome_test
);

# Populate the training arrays.
prepare_training_dataset();

# Build the conditional inference tree model.
build_model($counter_training);

# Populate the testing arrays.
prepare_testing_dataset();

# Check the performance of the model against the testing dataset.
check_model_performance($counter_testing);

###################################
# S  U  B  R  O  U  T  I  N  E  S #
###################################

# This function calls R and passes it all necessary information for it to build
# a conditional inference tree model from the data.
sub build_model
{
    my ($counter) = @_;

    # Initialise the R bridge object.
    my $R = Statistics::R->new;

    # R code for preparing the data frame.
    my $R_code = << "END";
    library(party)
    dataset <- data.frame(
        ID = rep(NA, $counter),
        T_pk_minus_T_ref = rep(NA, $counter),
        T_pk_minus_T_h = rep(NA, $counter),
        T_h_minus_T_ref = rep(NA, $counter),
        Outcome = rep(NA, $counter)
    )
END

    # Populate each row of the data frame with data from a single TPC.
    for ( 0 .. ( $counter - 1 ) )
    {
        my $r_counter = $_ + 1;

        $R_code .= << "END";
        dataset[$r_counter,] <- c(
            "$id[$_]",
            $t_pk_minus_t_ref[$_],
            $t_pk_minus_t_h[$_],
            $t_h_minus_t_ref[$_],
            "$outcome[$_]"
        )

        class(dataset\$T_pk_minus_T_ref) <- 'numeric'
        class(dataset\$T_pk_minus_T_h) <- 'numeric'
        class(dataset\$T_h_minus_T_ref) <- 'numeric'
        dataset\$Outcome <- factor(dataset\$Outcome, levels =
            c("below", "above"))
END
    }

    # Run the R code so far.
    $R->run($R_code);

    # R code that fits the model, generates the confusion matrix,
    # prepares a figure and saves the model as an R object file.
    $R_code = << "END";
    ctree_fit <- ctree(formula = as.factor(Outcome) ~ T_pk_minus_T_ref +
        T_pk_minus_T_h + T_h_minus_T_ref, data = dataset,
        control = ctree_control(
            mincriterion = 1 - 1e-10
        )
    )

    confusion_matrix <- table(dataset\$Outcome, predict(ctree_fit))

    TP <- as.numeric(confusion_matrix[2,2])
    FP <- as.numeric(confusion_matrix[1,2])
    TN <- as.numeric(confusion_matrix[1,1])
    FN <- as.numeric(confusion_matrix[2,1])

    MCC <- (TP * TN - FP * FN) / sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )

    pdf(file = "../Results/ctree_fit.pdf")
    plot(ctree_fit)
    dev.off()

    save.image(file = "../Results/ctree.rda")
END

    # Run the final piece of R code.
    $R->run($R_code);
    return 0;
}

# This function calculates the real B(T_ref) value from the parameters of the
# fit.
sub calc_b_t_ref
{
    my ( $b0, $e, $e_d, $t_pk ) = @_;

    return (
        $b0 / (
            1 + ( $e / ( $e_d - $e ) ) *
              exp( ( ($e_d) / $K ) * ( 1 / $t_pk - 1 / $T_REF ) )
        )
    );
}

# This function calculates the T_h value from T_pk, E_D, and E values.
sub calc_t_h
{
    my ( $t_pk, $e_d, $e ) = @_;

    return ( $t_pk * $e_d ) /
      ( $t_pk * $K * log( ( -$e ) / ( $e - $e_d ) ) + $e_d );
}

# This function checks if B0 >= P_pk, by checking if the left part of the
# following inequality is greater than the right part:
#
# 1 + E/(E_D - E) >= exp((-E/K) * (1/T_pk - 1/T_ref))
sub check_if_B0_greater_than_P_pk
{
    my ( $e, $e_d, $t_pk ) = @_;

    my $first_part = 1 + $e / ( $e_d - $e );
    my $second_part = exp( ( -$e / $K ) * ( 1 / $t_pk - 1 / $T_REF ) );

    if ( $first_part >= $second_part )
    {
        return 'above';
    }
    else
    {
        return 'below';
    }
}

# This function calls R and tests the performance of the previously built model
# on new data (testing dataset).
sub check_model_performance
{
    my ($counter) = @_;

    # Initialise the R bridge object.
    my $R = Statistics::R->new;

    # R code for preparing the data frame.
    my $R_code = << "END";
    library(party)
    load("../Results/ctree.rda")
    
    dataset <- data.frame(
        ID = rep(NA, $counter),
        T_pk_minus_T_ref = rep(NA, $counter),
        T_pk_minus_T_h = rep(NA, $counter),
        T_h_minus_T_ref = rep(NA, $counter),
        Outcome = rep(NA, $counter)
    )
END

    # Populate each row of the data frame with data from a single TPC.
    for ( 0 .. ( $counter - 1 ) )
    {
        my $r_counter = $_ + 1;

        $R_code .= << "END";
        dataset[$r_counter,] <- c(
            "$id_test[$_]",
            $t_pk_minus_t_ref_test[$_],
            $t_pk_minus_t_h_test[$_],
            $t_h_minus_t_ref_test[$_],
            "$outcome_test[$_]"
        )

        class(dataset\$T_pk_minus_T_ref) <- 'numeric'
        class(dataset\$T_pk_minus_T_h) <- 'numeric'
        class(dataset\$T_h_minus_T_ref) <- 'numeric'
        dataset\$Outcome <- factor(dataset\$Outcome, levels =
            c("below", "above"))
END
    }

    # Run the R code so far.
    $R->run($R_code);

    # R code that tests the predictions of the model against the new data.
    $R_code = << "END";
    confusion_matrix <- table(dataset\$Outcome, predict(ctree_fit, 
        newdata = dataset))

    TP <- as.numeric(confusion_matrix[2,2])
    FP <- as.numeric(confusion_matrix[1,2])
    TN <- as.numeric(confusion_matrix[1,1])
    FN <- as.numeric(confusion_matrix[2,1])

    MCC <- (TP * TN - FP * FN) / sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )
    
    save.image(file = "../Results/ctree_test.rda")
END

    # Run the final piece of R code.
    $R->run($R_code);
    return 0;
}

# This function goes through the testing dataset and calculates differences
# between the temperature variables (T_h, T_pk, T_ref) and whether B0 >= P_pk.
sub prepare_testing_dataset
{
    # For every fitted TPC ...
    foreach ( @{$testing_dataset} )
    {
        $counter_testing++;
        say "Now at $counter_testing ...";

        # Get the useful information for this TPC.
        my $species = $_->[0];
        my $ref     = $_->[1];
        my $b0      = $_->[2];
        my $e       = $_->[3];
        my $t_pk    = $_->[4];
        my $e_d     = $_->[5];

        # Calculate B(T_ref).
        my $b_t_ref = calc_b_t_ref( $b0, $e, $e_d, $t_pk + 273.15 );

        # Calculate T_h and convert it to °C.
        my $t_h = calc_t_h( $t_pk + 273.15, $e_d, $e );
        $t_h -= 273.15;

        # Ignore fits with B0 == 0 or B(T_ref) == 0 (just in case they exist).
        next if $b0 == 0 or $b_t_ref == 0;

        # Generate a unique ID for this TPC.
        push @id_test, $species . q{_} . $ref;

        # Store:
        # 1) T_pk - T_ref
        # 2) T_h - T_ref
        # 3) T_pk - T_h
        #
        # As T_ref is set at 0°C, T_pk - T_ref = T_pk and T_h - T_ref = T_h.
        push @t_pk_minus_t_ref_test, $t_pk;
        push @t_h_minus_t_ref_test,  $t_h;
        push @t_pk_minus_t_h_test,   $t_pk - $t_h;

        # Check if B0 >= P_pk.
        my $result = check_if_B0_greater_than_P_pk( $e, $e_d, $t_pk + 273.15 );
        push @outcome_test, $result;

    }
    return 0;
}

# This function goes through the training dataset and calculates differences
# between the temperature variables (T_h, T_pk, T_ref) and whether B0 >= P_pk.
sub prepare_training_dataset
{

    # For every fitted TPC ...
    foreach ( @{$training_dataset} )
    {
        $counter_training++;
        say "Now at $counter_training ...";

        # Get the useful information for this TPC.
        my $species = $_->[0];
        my $ref     = $_->[1];
        my $b0      = $_->[2];
        my $e       = $_->[3];
        my $t_pk    = $_->[4];
        my $e_d     = $_->[5];

        # Calculate B(T_ref).
        my $b_t_ref = calc_b_t_ref( $b0, $e, $e_d, $t_pk + 273.15 );

        # Calculate T_h and convert it to °C.
        my $t_h = calc_t_h( $t_pk + 273.15, $e_d, $e );
        $t_h -= 273.15;

        # Ignore fits with B0 == 0 or B(T_ref) == 0 (just in case they exist).
        next if $b0 == 0 or $b_t_ref == 0;

        # Generate a unique ID for this TPC.
        push @id, $species . q{_} . $ref;

        # Store:
        # 1) T_pk - T_ref
        # 2) T_h - T_ref
        # 3) T_pk - T_h
        #
        # As T_ref is set at 0°C, T_pk - T_ref = T_pk and T_h - T_ref = T_h.
        push @t_pk_minus_t_ref, $t_pk;
        push @t_h_minus_t_ref,  $t_h;
        push @t_pk_minus_t_h,   $t_pk - $t_h;

        # Check if B0 >= P_pk.
        my $result = check_if_B0_greater_than_P_pk( $e, $e_d, $t_pk + 273.15 );
        push @outcome, $result;

    }
    return 0;
}
